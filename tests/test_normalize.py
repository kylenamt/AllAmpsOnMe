from t3k import normalize


def test_text_coerces_scalars_and_objects():
    assert normalize.text("  amp ") == "amp"
    assert normalize.text({"name": "Amp"}) == "Amp"
    assert normalize.text({"slug": "full-rig"}) == "full-rig"
    assert normalize.text(2) == "2"
    assert normalize.text(None) == ""


def test_map_architecture():
    assert normalize.map_architecture(2) == "A2"
    assert normalize.map_architecture("2") == "A2"
    assert normalize.map_architecture("v2") == "A2"
    assert normalize.map_architecture(1) == "A1"
    assert normalize.map_architecture("A1") == "A1"
    assert normalize.map_architecture("wavenet") is None
    assert normalize.map_architecture(None) is None


def test_is_amp_gear_and_exclusions():
    assert normalize.is_amp_gear("amp")
    assert normalize.is_amp_gear({"slug": "amp"})
    assert not normalize.is_amp_gear("full-rig")
    assert normalize.should_exclude("Marshall Cab IR")
    assert not normalize.should_exclude("Marshall JCM800 lead")


def test_classify_gain_precedence():
    assert normalize.classify_gain(["high gain"], "Lead", "") == "high_gain"
    assert normalize.classify_gain(["crunch"], "JCM800", "") == "crunch"
    assert normalize.classify_gain([], "Deluxe Clean", "") == "clean"
    assert normalize.classify_gain([], "mystery box", "") == "unknown"


def test_normalize_tone_and_build_row_live_schema():
    tone = {
        "id": 42,
        "title": "Marshall JCM800 Lead",
        "url": "https://www.tone3000.com/t/42",
        "gear": {"slug": "amp"},
        "format": "nam",
        "license": {"name": "T3K"},
        "tags": [{"name": "high gain"}, {"name": "lead"}],
        "makes": [{"name": "Marshall"}],
        "user": {"id": "u9", "username": "bob"},
        "downloads_count": 250,
        "favorites_count": 12,
        "created_at": "2025-02-02T00:00:00Z",
    }
    model = {"id": 7, "name": "JCM800 Lead", "model_url": "https://x/7.nam",
             "architecture_version": 2, "tone_id": 42}

    tnorm = normalize.normalize_tone(tone)
    assert tnorm["tone_id"] == 42
    assert tnorm["creator"] == "bob"
    assert tnorm["downloads"] == 250
    assert tnorm["license"] == "T3K"

    row = normalize.build_candidate_row(tnorm, model)
    assert row["make"] == "marshall"
    assert row["architecture"] == "A2"
    assert row["gain_bucket"] == "high_gain"
    assert row["capture_type"] == "di"
    assert row["model_url"] == "https://x/7.nam"
    assert normalize.candidate_is_amp_nam(row)


def test_candidate_rejects_cab_and_bad_arch():
    tnorm = normalize.normalize_tone({"id": 1, "title": "Mesa Cab IR", "gear": "amp",
                                      "makes": ["Mesa"], "license": "T3K",
                                      "user": {"id": "u", "username": "x"}})
    row = normalize.build_candidate_row(tnorm, {"id": 1, "name": "Cabinet IR",
                                                "architecture_version": 2,
                                                "model_url": "https://x/1.nam"})
    assert not normalize.candidate_is_amp_nam(row)  # excluded by 'cab'/'ir' keyword
