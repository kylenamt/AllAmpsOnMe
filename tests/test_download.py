import hashlib
from pathlib import Path

import pandas as pd

from conftest import make_candidate

from t3k import download as dl
from t3k import manifest
from t3k.manifest import STATUS_DOWNLOADED, STATUS_SELECTED


class FakeClient:
    def __init__(self, fail_urls=(), a1_models=None):
        self.fail_urls = set(fail_urls)
        self.a1_models = a1_models or {}
        self.downloaded = []

    def download_model(self, url, dest):
        if url in self.fail_urls:
            raise RuntimeError("boom")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        data = b"data-" + url.encode()
        Path(dest).write_bytes(data)
        self.downloaded.append(url)
        return hashlib.sha256(data).hexdigest(), len(data)

    def list_models(self, tone_id, architecture=None):
        return self.a1_models.get((tone_id, architecture), [])


def _df(rows):
    return manifest.ensure_schema(pd.DataFrame(rows))


def test_download_success(settings):
    df = _df([make_candidate(tone_id=1, model_id=1, model_url="U1", status=STATUS_SELECTED)])
    client = FakeClient()
    out = dl.download(client, df, settings, sleep_fn=lambda s: None)
    row = out.iloc[0]
    assert row["status"] == STATUS_DOWNLOADED
    assert row["file_bytes"] == len(b"data-U1")
    assert Path(row["file_path"]).is_file()


def test_download_resumable_skips_existing(settings):
    dest = settings.captures_dir / "1" / "1.nam"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"already")
    sha = hashlib.sha256(b"already").hexdigest()
    df = _df([make_candidate(tone_id=1, model_id=1, model_url="U1",
                             status=STATUS_SELECTED, sha256=sha)])

    class NoDownload(FakeClient):
        def download_model(self, url, dest):  # must not be called
            raise AssertionError("should not download")

    out = dl.download(NoDownload(), df, settings, sleep_fn=lambda s: None)
    assert out.iloc[0]["status"] == STATUS_DOWNLOADED


def test_download_a2_to_a1_fallback(settings):
    df = _df([make_candidate(tone_id=5, model_id=50, model_url="A2URL",
                             architecture="A2", status=STATUS_SELECTED)])
    client = FakeClient(
        fail_urls=["A2URL"],
        a1_models={(5, 1): [{"id": 51, "name": "A1 sub", "model_url": "A1URL",
                             "architecture_version": 1}]},
    )
    out = dl.download(client, df, settings, sleep_fn=lambda s: None)
    row = out.iloc[0]
    assert row["status"] == STATUS_DOWNLOADED
    assert row["architecture"] == "A1"
    assert bool(row["architecture_fallback"]) is True
    assert row["model_id"] == 51
