"""The one ``openamp`` command-line entry point.

The whole pipeline is a flat list of verbs, in run order:

  acquire :  auth · discover · select · download · validate · dedup · finalize · status
  corpus  :  corpus · subset · render · verify
  emulate :  emulate · emulate-compare · emulate-demo · emulate-enroll

Each command is thin: it loads the one :class:`~openamp.core.config.Config`, then calls
into a stage module. Heavy imports (torch, NAM, soundfile) live inside the command
bodies so ``openamp --help`` stays fast, and stage modules are imported with a
``_mod`` suffix so they never shadow a verb of the same name.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from openamp.core import manifest as manifest_mod
from openamp.core.config import ConfigError, load_config


# --- Shared helpers ------------------------------------------------------------
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _config(config_path: Path | None = None, *, require_key: bool = False):
    try:
        return load_config(config_path, require_key=require_key)
    except ConfigError as exc:
        raise click.ClickException(str(exc))


def _client(cfg):
    from openamp.acquire.client import T3KClient
    return T3KClient(cfg)


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _load_working(cfg):
    df = manifest_mod.read_manifest(cfg.manifest_path)
    if df.empty:
        raise click.ClickException("Empty manifest. Run `openamp select` first.")
    return df


def _print_status(cfg) -> None:
    df = manifest_mod.read_manifest(cfg.manifest_path)
    counts = manifest_mod.counts_by_status(df)
    if not counts:
        click.echo("Manifest is empty.")
        return
    click.echo("Status counts:")
    for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])):
        click.echo(f"  {str(k):18s} {v}")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def cli(verbose: bool) -> None:
    """Open-Amp3000 pipeline: acquire TONE3000 captures, render a clean corpus,
    and train the FiLM-TCN amp foundation model.

    Run order: auth discover select download validate dedup finalize · corpus
    subset render verify · emulate emulate-compare emulate-demo emulate-enroll.
    """
    _setup_logging(verbose)


# =========================== Acquisition (TONE3000) ============================
@cli.command()
@click.option("--headless", is_flag=True, help="Print the URL and paste the redirect (no local browser).")
def auth(headless: bool) -> None:
    """Run the OAuth standard flow and verify with GET /user."""
    from openamp.acquire.auth import run_standard_flow

    cfg = _config(require_key=True)
    run_standard_flow(cfg, headless=headless)
    user = _client(cfg).get_user()
    username = user.get("username") or user.get("name") or user.get("id")
    click.secho(f"Authenticated as: {username}", fg="green")


@cli.command()
@click.option("--max-candidates", default=4000, show_default=True)
@click.option("--max-pages", default=20, show_default=True, help="Max pages per search pass.")
@click.option("--page-size", default=50, show_default=True)
@click.option("--no-resume", is_flag=True, help="Ignore existing candidates.parquet.")
def discover(max_candidates: int, max_pages: int, page_size: int, no_resume: bool) -> None:
    """Build the candidate model pool -> candidates.parquet."""
    from openamp.acquire import discover as discover_mod

    cfg = _config(require_key=True)
    df = discover_mod.discover(_client(cfg), cfg, max_candidates=max_candidates,
                               max_pages=max_pages, page_size=page_size, resume=not no_resume)
    click.secho(f"Candidates: {len(df)} models across "
                f"{df['tone_id'].nunique() if not df.empty else 0} tones "
                f"-> {cfg.candidates_path}", fg="green")


@cli.command()
@click.option("--target", default=None, type=int, help="Selection size (default 430).")
@click.option("--seed", default=None, type=int)
def select(target, seed) -> None:
    """Diversity selection -> manifest.parquet (status=selected)."""
    from openamp.acquire import select as select_mod

    cfg = _config()
    candidates = manifest_mod.read_manifest(cfg.candidates_path, manifest_mod.DISCOVERY_COLUMNS)
    if candidates.empty:
        raise click.ClickException("No candidates found. Run `openamp discover` first.")
    selected = select_mod.select(candidates, cfg, target=target, seed=seed)
    manifest_mod.write_manifest(selected, cfg.manifest_path)
    click.echo(select_mod.format_summary(select_mod.summarize(selected)))
    click.secho(f"\nWrote {len(selected)} selected -> {cfg.manifest_path}", fg="green")


@cli.command()
def download() -> None:
    """Download selected captures -> data/captures/."""
    from openamp.acquire import download as download_mod

    cfg = _config(require_key=True)
    download_mod.download(_client(cfg), _load_working(cfg), cfg)
    _print_status(cfg)


@cli.command()
def validate() -> None:
    """Load + render-probe each downloaded capture."""
    from openamp.acquire import validate as validate_mod

    cfg = _config()
    validate_mod.validate(_load_working(cfg), cfg)
    _print_status(cfg)


@cli.command()
def dedup() -> None:
    """Remove exact-hash and near-duplicate (ESR) captures."""
    from openamp.acquire import dedup as dedup_mod

    cfg = _config()
    dedup_mod.dedup(_load_working(cfg), cfg)
    _print_status(cfg)


@cli.command()
def finalize() -> None:
    """Assign device_ids, write final manifest + rejected."""
    from openamp.acquire import finalize as finalize_mod

    cfg = _config()
    final_df, rejected_df = finalize_mod.finalize(_load_working(cfg), cfg)
    click.echo(finalize_mod.summarize(final_df))
    click.secho(f"\nFinal: {len(final_df)} -> {cfg.manifest_path}", fg="green")
    click.secho(f"Rejected: {len(rejected_df)} -> {cfg.rejected_path}", fg="yellow")


@cli.command("migrate-a2")
@click.option("--limit", default=None, type=int, help="Migrate only the first N devices.")
def migrate_a2(limit) -> None:
    """Re-point finalized devices at their A2 captures (device_ids preserved)."""
    from openamp.acquire import migrate as migrate_mod

    cfg = _config(require_key=True)
    final = manifest_mod.read_manifest(cfg.manifest_path, manifest_mod.FINAL_COLUMNS)
    if final.empty:
        raise click.ClickException("No final manifest. Run `openamp finalize` first.")
    migrated = migrate_mod.migrate(_client(cfg), final, cfg, limit=limit)
    click.echo(migrate_mod.summarize(migrated))
    click.secho(f"\nMigrated manifest -> {cfg.manifest_path}", fg="green")


@cli.command()
def status() -> None:
    """Show manifest status counts."""
    _print_status(_config())


# ============================ Corpus + rendering ===============================
@cli.command()
@click.option("--config", "config_path", type=Path, default=None, help="configs/openamp.yaml")
@click.option("--force", is_flag=True, help="Rebuild even if outputs are current.")
def corpus(config_path, force) -> None:
    """Raw sources -> 48 kHz clean corpus + splits + clip grid."""
    from openamp.corpus import build as corpus_mod

    cfg = _config(config_path)
    try:
        df = corpus_mod.prepare(cfg, force=force)
    except corpus_mod.CorpusInputError as exc:
        raise click.ClickException(str(exc))
    click.secho(f"Corpus: {len(df)} files -> {cfg.corpus_manifest_path}", fg="green")


@cli.command()
@click.option("--size", default=450, show_default=True, help="Target number of devices to render.")
@click.option("--min-fraction", default=0.15, show_default=True, help="Per-gain-bucket floor.")
@click.option("--cap-tone", default=4, show_default=True, help="Max devices per tone upload.")
@click.option("--cap-creator", default=12, show_default=True, help="Max devices per creator.")
@click.option("--cap-makemodel", default=6, show_default=True, help="Max devices per (make, model).")
@click.option("--write/--no-write", default=True, help="Persist the id list for render.")
@click.option("--config", "config_path", type=Path, default=None)
def subset(size, min_fraction, cap_tone, cap_creator, cap_makemodel, write, config_path) -> None:
    """Pick a diverse, gain-balanced subset of devices to render.

    Trims the (often larger) acquisition manifest to a diverse subset (caps + gain
    floor) without touching it. Feed the result to ``render --devices @<path>``.
    """
    from openamp.corpus import subset as subset_mod

    cfg = _config(config_path)
    manifest = manifest_mod.read_manifest(cfg.manifest_path, subset_mod.SUBSET_COLUMNS)
    if manifest.empty:
        raise click.ClickException(f"Manifest not found or empty: {cfg.manifest_path}")

    caps = (cap_tone, cap_creator, cap_makemodel)
    ids = subset_mod.select_render_subset(
        manifest, size, min_fraction=min_fraction,
        cap_tone=cap_tone, cap_creator=cap_creator, cap_makemodel=cap_makemodel)
    summary = subset_mod.summarize(manifest, ids)
    click.echo(subset_mod.format_summary(summary, size))
    if len(ids) < size:
        click.secho(f"note: caps exhausted the pool at {len(ids)} (< {size}); "
                    "raise --cap-tone / --cap-creator to grow it.", fg="yellow")
    if write:
        subset_mod.write_subset_file(cfg.render_subset_path, ids, summary,
                                     target=size, caps=caps, min_fraction=min_fraction)
        click.secho(f"Wrote {len(ids)} device ids -> {cfg.render_subset_path}", fg="green")
        click.echo(f"Render them with:  openamp render --devices @{cfg.render_subset_path}")


@cli.command()
@click.option("--devices", default=None, help='Device subset, e.g. "0-9", "0,2,5", or @file.')
@click.option("--dry-run", is_flag=True, help="List work without rendering.")
@click.option("--device", default="cuda", help="Torch device (cuda/cpu).")
@click.option("--io-workers", default=0, show_default=True,
              help="Threads for the FLAC encode/hash pass (0 = auto); overlaps the GPU.")
@click.option("--config", "config_path", type=Path, default=None)
def render(devices, dry_run, device, io_workers, config_path) -> None:
    """GPU-render the corpus through selected devices (resumable)."""
    from openamp.corpus import render as render_mod

    cfg = _config(config_path)
    df = render_mod.run(cfg, devices=devices, dry_run=dry_run, device=device,
                        io_workers=io_workers)
    click.secho(f"Renders: {len(df)} rows -> {cfg.renders_manifest_path}", fg="green")


@cli.command()
@click.option("--config", "config_path", type=Path, default=None)
def verify(config_path) -> None:
    """Completeness + sanity checks + QA exports + report -> devices_final.parquet."""
    from openamp.corpus import verify as verify_mod

    cfg = _config(config_path)
    finals = verify_mod.verify(cfg)
    click.secho(f"Final devices: {len(finals)} -> {cfg.devices_final_path}", fg="green")
    click.secho(f"Report: {cfg.render_report_path}", fg="green")


# ===================== Emulation foundation model (FiLM-TCN) ===================
def _emulate_name(config_path: Path | None) -> str:
    """Run name = config file stem (``configs/emulate/<name>.yaml``), else 'default'."""
    return Path(config_path).stem if config_path else "default"


@cli.command()
@click.option("--config", "config_path", type=Path, default=None,
              help="configs/emulate/<name>.yaml; run name is the file stem.")
@click.option("--device", default="cuda")
@click.option("--epochs", type=int, default=None, help="Override max epochs.")
@click.option("--resume", is_flag=True, help="Continue from the run's last.pt; the "
              "config is re-read and its training knobs (lr, weight_decay, plateau_*, "
              "epochs, losses) take effect — structural knobs must not change.")
@click.option("--overfit", is_flag=True, help="Sanity #1: overfit one batch, print ESR.")
@click.option("--limit-devices", type=int, default=None,
              help="Mini-run on the first N render-ok devices (sanity #2).")
def emulate(config_path, device, epochs, resume, overfit, limit_devices) -> None:
    """Train the FiLM-conditioned TCN amp foundation model (one-to-many)."""
    from openamp.emulate import train as emu_train

    cfg = _config(config_path)
    name = _emulate_name(config_path)
    if overfit:
        emu_train.overfit_one_batch(cfg, name=name, device=device)
        return
    emu_train.train(cfg, name=name, device=device, epochs=epochs, resume=resume,
                    limit_devices=limit_devices)


@cli.command("emulate-compare")
@click.argument("runs", nargs=-1, type=Path)
@click.option("--device", default=None, help="cuda/cpu (default: auto).")
@click.option("--n-pairs", type=int, default=None, help="Test pairs per run (default: config).")
def emulate_compare(runs, device, n_pairs) -> None:
    """Evaluate emulation runs on the test split -> results/emulate/comparison.csv."""
    from openamp.emulate import evaluate as emu_eval

    cfg = _config()
    run_dirs = [Path(r) for r in runs] or sorted(
        p.parent for p in cfg.emulate_dir.glob("*/checkpoint.pt"))
    if not run_dirs:
        raise click.ClickException(f"No runs found under {cfg.emulate_dir}.")
    rows = emu_eval.compare(cfg, run_dirs, device=device or _default_device(), n_pairs=n_pairs)
    for r in rows:
        click.echo(f"  {r['run_name']:20s} ESR={r['test_ESR_mean']:.4f} "
                   f"MRSL={r['test_MRSL_mean']:.3f} params={r['params']:,}")


@cli.command("emulate-demo")
@click.argument("run", type=Path)
@click.option("--n-devices", default=4, show_default=True)
@click.option("--seconds", default=10.0, show_default=True)
@click.option("--device", default=None, help="cuda/cpu (default: auto).")
def emulate_demo(run, n_devices, seconds, device) -> None:
    """Export clean/target/prediction WAVs for a run (the listening check)."""
    from openamp.emulate import evaluate as emu_eval

    cfg = _config()
    emu_eval.export_demos(cfg, run, n_devices=n_devices, seconds=seconds,
                          device=device or _default_device())


@cli.command("emulate-enroll")
@click.argument("run", type=Path)
@click.option("--devices", default=None,
              help="Comma-separated device ids (default: the run's holdout set).")
@click.option("--pairs", default=1000, show_default=True,
              help="Training pairs per device per epoch (optimization budget).")
@click.option("--epochs", default=30, show_default=True, help="Max epochs (early-stopped).")
@click.option("--lr", default=1e-2, show_default=True)
@click.option("--device", default=None, help="cuda/cpu (default: auto).")
@click.option("--seed", type=int, default=None, help="Override the config seed.")
def emulate_enroll(run, devices, pairs, epochs, lr, device, seed) -> None:
    """Enroll unseen devices: freeze a trained run, fit new embeddings only."""
    from openamp.emulate import enroll as emu_enroll

    cfg = _config()
    if not (run / "checkpoint.pt").is_file():
        raise click.ClickException(f"No checkpoint.pt under {run}.")
    ids = [int(x) for x in devices.split(",") if x.strip()] if devices else None
    m = emu_enroll.enroll(cfg, run, device_ids=ids, pairs=pairs, epochs=epochs, lr=lr,
                          device=device or _default_device(), seed=seed)
    click.echo(f"  enrolled {m['n_enrolled']} devices: test_ESR={m['test_esr_mean']} "
               f"(baseline {m['baseline_test_esr_mean']}, "
               f"trained {m['trained_test_esr_mean']})")


def main() -> None:
    try:
        cli()
    except KeyboardInterrupt:  # pragma: no cover
        click.secho("\nInterrupted.", fg="red")
        sys.exit(130)


if __name__ == "__main__":
    main()
