import pandas as pd

from t3k import manifest


def test_read_missing_returns_empty_schema(tmp_path):
    df = manifest.read_manifest(tmp_path / "nope.parquet")
    assert list(df.columns) == manifest.ALL_COLUMNS
    assert df.empty


def test_write_read_roundtrip_with_list_column(tmp_path):
    df = manifest.empty_manifest()
    df = pd.concat([df, pd.DataFrame([{
        "tone_id": 1, "model_id": 2, "tags": ["clean", "fender"], "make": "fender",
        "status": manifest.STATUS_SELECTED,
    }])], ignore_index=True)
    df = manifest.ensure_schema(df)
    path = tmp_path / "m.parquet"
    manifest.write_manifest(df, path)
    back = manifest.read_manifest(path)
    assert list(back.loc[0, "tags"]) == ["clean", "fender"]
    assert back.loc[0, "status"] == manifest.STATUS_SELECTED


def test_ensure_schema_adds_missing_columns():
    df = pd.DataFrame([{"tone_id": 1}])
    out = manifest.ensure_schema(df)
    for col in manifest.ALL_COLUMNS:
        assert col in out.columns


def test_set_status_and_counts():
    df = manifest.ensure_schema(pd.DataFrame([{"model_id": i} for i in range(4)]))
    manifest.set_status(df, df["model_id"] < 2, manifest.STATUS_DOWNLOADED, sha256="abc")
    manifest.set_status(df, df["model_id"] >= 2, manifest.STATUS_REJECTED)
    counts = manifest.counts_by_status(df)
    assert counts[manifest.STATUS_DOWNLOADED] == 2
    assert counts[manifest.STATUS_REJECTED] == 2
    assert (df.loc[df["model_id"] < 2, "sha256"] == "abc").all()
