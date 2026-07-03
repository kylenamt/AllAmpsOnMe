"""`t3k` command-line entry point (spec §5).

Run order:  auth -> discover -> select -> download -> validate -> dedup -> finalize
Each subcommand reads/writes the parquet manifests and is independently
re-runnable (idempotent).
"""

from __future__ import annotations

import logging
import sys

import click

from . import discover as discover_mod
from . import dedup as dedup_mod
from . import download as download_mod
from . import finalize as finalize_mod
from . import manifest as manifest_mod
from . import select as select_mod
from . import validate as validate_mod
from .auth import run_standard_flow
from .client import T3KClient
from .config import ConfigError, load_settings


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _settings(require_key: bool = True):
    try:
        return load_settings(require_key=require_key)
    except ConfigError as exc:
        raise click.ClickException(str(exc))


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """TONE3000 amp-capture acquisition (Phase 1)."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# --- auth ----------------------------------------------------------------------
@cli.command()
@click.option("--headless", is_flag=True, help="Print the URL and paste the redirect (no local browser).")
def auth(headless: bool) -> None:
    """Run the OAuth standard flow and verify with GET /user."""
    settings = _settings()
    run_standard_flow(settings, headless=headless)
    client = T3KClient(settings)
    user = client.get_user()
    username = user.get("username") or user.get("name") or user.get("id")
    click.secho(f"Authenticated as: {username}", fg="green")


# --- discover ------------------------------------------------------------------
@cli.command()
@click.option("--max-candidates", default=4000, show_default=True)
@click.option("--max-pages", default=20, show_default=True, help="Max pages per search pass.")
@click.option("--page-size", default=50, show_default=True)
@click.option("--no-resume", is_flag=True, help="Ignore existing candidates.parquet.")
def discover(max_candidates: int, max_pages: int, page_size: int, no_resume: bool) -> None:
    """Build the candidate model pool -> candidates.parquet."""
    settings = _settings()
    client = T3KClient(settings)
    df = discover_mod.discover(client, settings, max_candidates=max_candidates,
                               max_pages=max_pages, page_size=page_size, resume=not no_resume)
    click.secho(f"Candidates: {len(df)} models across "
                f"{df['tone_id'].nunique() if not df.empty else 0} tones "
                f"-> {settings.candidates_path}", fg="green")


# --- select --------------------------------------------------------------------
@cli.command()
@click.option("--target", default=None, type=int, help="Selection size (default 430).")
@click.option("--seed", default=None, type=int)
def select(target, seed) -> None:
    """Diversity selection -> manifest.parquet (status=selected)."""
    settings = _settings(require_key=False)
    candidates = manifest_mod.read_manifest(settings.candidates_path, manifest_mod.DISCOVERY_COLUMNS)
    if candidates.empty:
        raise click.ClickException("No candidates found. Run `t3k discover` first.")
    selected = select_mod.select(candidates, settings, target=target, seed=seed)
    manifest_mod.write_manifest(selected, settings.manifest_path)
    click.echo(select_mod.format_summary(select_mod.summarize(selected)))
    click.secho(f"\nWrote {len(selected)} selected -> {settings.manifest_path}", fg="green")


# --- download ------------------------------------------------------------------
@cli.command()
def download() -> None:
    """Download selected captures -> data/captures/."""
    settings = _settings()
    client = T3KClient(settings)
    df = _load_working(settings)
    download_mod.download(client, df, settings)
    _print_status(settings)


# --- validate ------------------------------------------------------------------
@cli.command()
def validate() -> None:
    """Load + render-probe each downloaded capture."""
    settings = _settings(require_key=False)
    client = None
    try:
        client = T3KClient(_settings())  # enables A2->A1 fallback if authed
    except click.ClickException:
        pass
    df = _load_working(settings)
    validate_mod.validate(df, settings, client=client)
    _print_status(settings)


# --- dedup ---------------------------------------------------------------------
@cli.command()
def dedup() -> None:
    """Remove exact-hash and near-duplicate (ESR) captures."""
    settings = _settings(require_key=False)
    df = _load_working(settings)
    dedup_mod.dedup(df, settings)
    _print_status(settings)


# --- finalize ------------------------------------------------------------------
@cli.command()
@click.option("--no-top-up", is_flag=True, help="Do not auto-select replacements to reach quota.")
def finalize(no_top_up: bool) -> None:
    """Top up to quota, assign device_ids, write final manifest + rejected."""
    settings = _settings(require_key=False)
    df = _load_working(settings)
    candidates = manifest_mod.read_manifest(settings.candidates_path, manifest_mod.DISCOVERY_COLUMNS)

    top_up_fn = None
    if not no_top_up and not candidates.empty:
        try:
            client = T3KClient(_settings())

            def top_up_fn(work_df):  # noqa: E306
                work_df = download_mod.download(client, work_df, settings)
                work_df = validate_mod.validate(work_df, settings, client=client)
                work_df = dedup_mod.dedup(work_df, settings)
                return work_df
        except click.ClickException:
            click.secho("No API key: skipping top-up loop.", fg="yellow")

    final_df, rejected_df = finalize_mod.finalize(df, settings, candidates=candidates,
                                                  top_up_fn=top_up_fn)
    click.echo(finalize_mod.summarize(final_df))
    click.secho(f"\nFinal: {len(final_df)} -> {settings.manifest_path}", fg="green")
    click.secho(f"Rejected: {len(rejected_df)} -> {settings.rejected_path}", fg="yellow")


# --- status --------------------------------------------------------------------
@cli.command()
def status() -> None:
    """Show manifest status counts."""
    settings = _settings(require_key=False)
    _print_status(settings)


# --- helpers -------------------------------------------------------------------
def _load_working(settings):
    df = manifest_mod.read_manifest(settings.manifest_path)
    if df.empty:
        raise click.ClickException("Empty manifest. Run `t3k select` first.")
    return df


def _print_status(settings) -> None:
    df = manifest_mod.read_manifest(settings.manifest_path)
    counts = manifest_mod.counts_by_status(df)
    if not counts:
        click.echo("Manifest is empty.")
        return
    click.echo("Status counts:")
    for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])):
        click.echo(f"  {str(k):18s} {v}")


def main() -> None:
    try:
        cli(obj={})
    except KeyboardInterrupt:  # pragma: no cover
        click.secho("\nInterrupted.", fg="red")
        sys.exit(130)


if __name__ == "__main__":
    main()
