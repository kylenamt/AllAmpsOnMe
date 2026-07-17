from openamp.acquire import discover as disc


class FakeClient:
    """Records the architecture filter each endpoint was called with."""

    def __init__(self, models):
        self.models = models
        self.search_architectures = []
        self.list_architectures = []

    def iter_search(self, *, max_pages=20, page_size=50, **filters):
        self.search_architectures.append(filters.get("architecture"))
        yield {
            "id": 1, "title": "Marshall JCM800", "gear": "amp",
            "url": "https://t3k/tones/1", "tags": ["high gain"],
            "makes": ["Marshall"], "user": {"id": "u1", "username": "someone"},
            "downloads_count": 10, "favorites_count": 1, "created_at": "2026-01-01",
        }

    def list_models(self, tone_id, architecture=None):
        self.list_architectures.append(architecture)
        return self.models


def _model(model_id, architecture_version):
    return {
        "id": model_id, "tone_id": 1, "name": "JCM800 Lead",
        "model_url": f"https://t3k/models/{model_id}.nam",
        "architecture_version": architecture_version, "sample_rate": 48000,
    }


def test_architecture_is_passed_to_the_models_endpoint_too(settings):
    """GET /models defaults to A1: omitting the filter there silently harvests A1
    captures out of tones the search picked for having A2."""
    client = FakeClient([_model(10, "2")])

    disc.discover(client, settings, sorts=("trending",), terms=(), max_pages=1)

    assert client.search_architectures == [2]
    assert client.list_architectures == [2]


def test_architecture_follows_the_config(settings):
    settings.architecture = 1
    client = FakeClient([_model(10, "1")])

    df = disc.discover(client, settings, sorts=("trending",), terms=(), max_pages=1)

    assert client.list_architectures == [1]
    assert df.iloc[0]["architecture"] == "A1"


def test_discovered_a2_models_are_labelled_a2(settings):
    client = FakeClient([_model(10, "2")])

    df = disc.discover(client, settings, sorts=("trending",), terms=(), max_pages=1)

    assert len(df) == 1
    assert df.iloc[0]["architecture"] == "A2"
    assert df.iloc[0]["model_id"] == 10
