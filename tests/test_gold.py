import json
from datetime import UTC, date, datetime

import pytest
from helpers import seeded_silver, utc_ms

from reviews_pipeline import gold, silver

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
PS5, XBOX = "playstation-5", "xbox-series-x"


@pytest.fixture(autouse=True)
def every_band_is_capped(monkeypatch):
    """Tests treat any band with rows as capped, like the 10,000-result queries; some override."""
    monkeypatch.setattr(gold, "CAPPED_ROWS", 1)


def test_coverage_window_start_is_latest_band_start():
    con = seeded_silver([
        (PS5, 2, "2026-11-21", utc_ms(2026, 11, 21, 21), "n"),   # negative band starts here
        (PS5, 9, "2026-11-22", utc_ms(2026, 11, 22, 2), "p"),    # positive band starts later
        (PS5, 6, "2026-11-20", utc_ms(2026, 11, 20, 5), "m"),    # neutral band, earliest
    ])
    cov = gold.coverage(con)[PS5]
    assert cov.window_start == datetime(2026, 11, 22, 2, tzinfo=UTC)
    assert cov.complete_from == date(2026, 11, 23)


def test_coverage_ignores_a_band_with_no_rows():
    con = seeded_silver([(PS5, 9, "2026-11-22", utc_ms(2026, 11, 22, 2), "p")])
    assert gold.coverage(con)[PS5].window_start == datetime(2026, 11, 22, 2, tzinfo=UTC)


def test_coverage_empty_silver_is_none():
    con = seeded_silver([(XBOX, 9, "2026-11-22", utc_ms(2026, 11, 22, 2), "p")])
    assert gold.coverage(con)[PS5] == gold.Coverage(None, None)


def test_valid_days_stops_at_today():
    assert gold.valid_days(NOW) == date(2026, 11, 24)


def day_rows(platform, day, scores, texts=None):
    """Rows for one day, versioned at noon UTC of that day."""
    y, m, d = map(int, day.split("-"))
    texts = texts or [f"unique {day} {i}" for i in range(len(scores))]
    return [(platform, s, day, utc_ms(y, m, d, 12), t) for s, t in zip(scores, texts, strict=True)]


def only(rows_by_platform, platform=PS5):
    return rows_by_platform[platform]


def test_daily_counts_bands_mean_and_median():
    con = seeded_silver(day_rows(PS5, "2026-11-23", [0, 1, 5, 9, 10]))
    (row,) = only(gold.daily(con, NOW))
    assert row["date"] == "2026-11-23"
    assert (row["n"], row["low"], row["mid"], row["high"]) == (5, 2, 1, 2)
    assert (row["mean"], row["median"]) == (5.0, 5)


def test_daily_repeated_share_counts_normalized_duplicates():
    con = seeded_silver(day_rows(PS5, "2026-11-23", [5, 5, 5, 5], ["Fake  ", "fake", "unique", ""]))
    (row,) = only(gold.daily(con, NOW))
    assert row["repeated"] == round(2 / 4, 4)  # two "fake"; the empty text never counts


def test_daily_marks_days_before_window_as_incomplete():
    rows = (
        day_rows(PS5, "2026-11-22", [9, 2, 6])
        + day_rows(PS5, "2026-11-23", [9, 2, 6])
    )
    con = seeded_silver(rows)  # every band starts 2026-11-22 12:00, so complete from the 23rd
    flags = {r["date"]: r["complete"] for r in only(gold.daily(con, NOW))}
    assert flags == {"2026-11-22": False, "2026-11-23": True}


def test_daily_excludes_future_junk_dates():
    rows = (
        day_rows(PS5, "2026-11-23", [5])
        + day_rows(PS5, "2026-11-24", [5])   # today: kept
        + day_rows(PS5, "2026-11-25", [5])   # tomorrow (time zone or junk): excluded
        + day_rows(PS5, "2027-11-21", [5])
    )
    days = [r["date"] for r in only(gold.daily(seeded_silver(rows), NOW))]
    assert days == ["2026-11-23", "2026-11-24"]


def test_daily_platform_without_rows_is_an_empty_list():
    out = gold.daily(seeded_silver(day_rows(PS5, "2026-11-23", [5])), NOW)
    assert out[XBOX] == []
    assert set(out) == {PS5, XBOX, "all"}
    assert out["all"] == out[PS5]


def at(day, hour, platform=PS5, score=5, text=None):
    y, m, d = map(int, day.split("-"))
    return (platform, score, day, utc_ms(y, m, d, hour), text or f"t {day} {hour} {score}")


def test_hourly_counts_only_complete_hours():
    con = seeded_silver([
        at("2026-11-23", 9, score=9),                      # before the window: dropped
        at("2026-11-23", 10, score=9), at("2026-11-23", 10, score=2),
        at("2026-11-23", 10, score=6),
        at("2026-11-23", 11, score=5), at("2026-11-23", 11, score=5, text="other"),
    ])
    rows = gold.hourly(con, NOW)[PS5]
    assert rows == [
        {"hour": "2026-11-23T10:00:00Z", "n": 3},
        {"hour": "2026-11-23T11:00:00Z", "n": 2},
    ]


def test_hourly_never_includes_the_future():
    con = seeded_silver([
        at("2026-11-24", 9, score=9), at("2026-11-24", 9, score=2), at("2026-11-24", 9, score=6),
        at("2026-11-24", 15, score=5),  # after NOW (12:00)
    ])
    assert [r["hour"] for r in gold.hourly(con, NOW)[PS5]] == ["2026-11-24T09:00:00Z"]


def test_meta_freshness_fields():
    rows = [
        at("2026-11-23", 10), at("2026-11-24", 11),
        at("2026-11-24", 20),                       # version after NOW: not "newest"
        ("playstation-5", 5, "2027-11-21", utc_ms(2026, 11, 24, 8), "junk"),
    ]
    con = seeded_silver(rows, seen_at=datetime(2026, 11, 24, 11, 30))
    meta = gold.meta(con, NOW)
    assert meta["generated_at"] == "2026-11-24T12:00:00Z"
    assert meta["last_success_at"] == "2026-11-24T11:30:00Z"
    assert meta["newest_review_at"] == "2026-11-24T11:00:00Z"
    assert (meta["rows"], meta["excluded_future_dates"]) == (4, 1)
    assert meta["coverage"][XBOX] == {"window_start": None, "complete_from": None}


def test_meta_on_empty_silver_has_nulls_and_zero_rows():
    meta = gold.meta(seeded_silver([]), NOW)
    assert meta["last_success_at"] is None and meta["newest_review_at"] is None
    assert (meta["rows"], meta["excluded_future_dates"]) == (0, 0)
    assert meta["drift"] == {PS5: None, XBOX: None, "all": None}


def drift_rows(latest_low):
    """Three earlier days with 10% low scores, then a latest day with `latest_low` low scores."""
    steady = ([0] + [5] * 8 + [9]) * 3
    rows = day_rows(PS5, "2026-11-18", [2, 6, 9])  # opens every band, so 19th onwards is complete
    for day in ("2026-11-19", "2026-11-20", "2026-11-21"):
        rows += day_rows(PS5, day, steady)
    return rows + day_rows(PS5, "2026-11-22", ([0] * (3 * latest_low) + [5] * (3 * (9 - latest_low)) + [9] * 3))


def test_drift_flags_a_large_move_in_a_band_share():
    drift = gold.meta(seeded_silver(drift_rows(6)), NOW)["drift"][PS5]
    assert drift == {"date": "2026-11-22", "metric": "low", "delta_pp": 50.0}


def test_drift_ignores_the_current_unfinished_day():
    today = day_rows(PS5, "2026-11-24", [0] * 10)  # NOW is 2026-11-24 12:00: day not over
    drift = gold.meta(seeded_silver(drift_rows(6) + today), NOW)["drift"][PS5]
    assert drift == {"date": "2026-11-22", "metric": "low", "delta_pp": 50.0}


def test_drift_ignores_a_move_within_the_threshold():
    assert gold.meta(seeded_silver(drift_rows(2)), NOW)["drift"][PS5] is None


def test_drift_is_null_with_fewer_than_three_prior_days():
    rows = day_rows(PS5, "2026-11-18", [2, 6, 9]) + day_rows(PS5, "2026-11-19", [0] * 9 + [9])
    assert gold.meta(seeded_silver(rows), NOW)["drift"][PS5] is None


def test_build_has_the_three_documents():
    out = gold.build(seeded_silver(day_rows(PS5, "2026-11-23", [5])), NOW)
    assert set(out) == {"meta", "daily", "hourly"}
    assert set(out["daily"]["platforms"]) == {PS5, XBOX, "all"}
    assert set(out["hourly"]["platforms"]) == {PS5, XBOX, "all"}


DAILY_KEYS = {"date", "n", "mean", "median", "low", "mid", "high", "repeated", "complete"}


def test_write_creates_three_valid_json_files(tmp_path):
    con = seeded_silver(day_rows(PS5, "2026-11-23", [5, 9]))
    gold.write(con, tmp_path / "out", NOW)
    for name in ("meta", "daily", "hourly"):
        assert isinstance(json.loads((tmp_path / "out" / f"{name}.json").read_text()), dict)


def test_written_json_contains_no_review_level_data(tmp_path):
    rows = day_rows(PS5, "2026-11-23", [5, 9], ["SECRET-TEXT-one", "SECRET-TEXT-two"])
    gold.write(seeded_silver(rows), tmp_path, NOW)
    for path in tmp_path.glob("*.json"):
        content = path.read_text()
        for needle in ("SECRET-TEXT", "hash-", "id-0", "id-1"):
            assert needle not in content, (path.name, needle)
    daily_rows = json.loads((tmp_path / "daily.json").read_text())["platforms"][PS5]
    assert {frozenset(row) for row in daily_rows} == {frozenset(DAILY_KEYS)}


def test_write_on_empty_silver_gives_valid_empty_files(tmp_path):
    gold.write(seeded_silver([]), tmp_path, NOW)
    assert json.loads((tmp_path / "daily.json").read_text()) == {
        "platforms": {PS5: [], XBOX: [], "all": []}
    }
    assert json.loads((tmp_path / "meta.json").read_text())["last_success_at"] is None


def test_main_exits_one_without_state_files(tmp_path):
    assert gold.main(["--state", str(tmp_path), "--out", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()


def test_main_writes_files_from_saved_silver(tmp_path):
    silver.save(seeded_silver(day_rows(PS5, "2026-11-23", [5, 9])), tmp_path / "state")
    out = tmp_path / "out"
    assert gold.main(["--state", str(tmp_path / "state"), "--out", str(out)]) == 0
    assert (out / "daily.json").exists()


def test_hourly_leaves_out_the_hour_still_being_collected():
    base = [at("2026-11-23", 10, score=9), at("2026-11-23", 10, score=2), at("2026-11-23", 10, score=6)]
    rows = base + [at("2026-11-24", 11, score=5), ("playstation-5", 5, "2026-11-24", utc_ms(2026, 11, 24, 12) + 5 * 60_000, "x")]
    con = seeded_silver(rows, seen_at=datetime(2026, 11, 24, 12, 7))  # run at 12:07
    hours = [r["hour"] for r in gold.hourly(con, datetime(2026, 11, 24, 12, 10, tzinfo=UTC))[PS5]]
    assert hours == ["2026-11-23T10:00:00Z", "2026-11-24T11:00:00Z"]


def test_daily_marks_today_as_incomplete_even_with_full_coverage():
    rows = day_rows(PS5, "2026-11-22", [9, 2, 6]) + day_rows(PS5, "2026-11-23", [9, 2, 6]) + day_rows(PS5, "2026-11-24", [9, 2, 6])
    flags = {r["date"]: r["complete"] for r in only(gold.daily(seeded_silver(rows), NOW))}
    assert flags == {"2026-11-22": False, "2026-11-23": True, "2026-11-24": False}


def test_repeated_texts_are_found_across_platforms():
    rows = day_rows(PS5, "2026-11-23", [5], ["same copy-pasted text"]) + day_rows(XBOX, "2026-11-23", [5], ["Same copy-pasted  text"])
    out = gold.daily(seeded_silver(rows), NOW)
    assert out[PS5][0]["repeated"] == 1.0 and out[XBOX][0]["repeated"] == 1.0


def test_coverage_uncapped_platform_is_complete_from_its_first_day(monkeypatch):
    monkeypatch.setattr(gold, "CAPPED_ROWS", 100)  # no band comes near the cap
    con = seeded_silver([
        (XBOX, 9, "2026-11-20", utc_ms(2026, 11, 20, 20), "a"),
        (XBOX, 2, "2026-11-21", utc_ms(2026, 11, 21, 8), "b"),
        (XBOX, 6, "2026-11-22", utc_ms(2026, 11, 22, 9), "c"),
    ])
    cov = gold.coverage(con)[XBOX]
    assert cov.window_start == datetime(2026, 11, 20, 20, tzinfo=UTC)
    assert cov.complete_from == date(2026, 11, 20)


def test_coverage_uses_only_the_capped_bands(monkeypatch):
    monkeypatch.setattr(gold, "CAPPED_ROWS", 3)
    rows = [(PS5, 2, "2026-11-22", utc_ms(2026, 11, 22, 1 + i), f"n{i}") for i in range(3)]   # negative: capped
    rows += [(PS5, 9, "2026-11-22", utc_ms(2026, 11, 22, 20), "p"), (PS5, 6, "2026-11-19", utc_ms(2026, 11, 19, 5), "m")]  # uncapped bands: ignored
    cov = gold.coverage(seeded_silver(rows))[PS5]
    assert cov.window_start == datetime(2026, 11, 22, 1, tzinfo=UTC)
    assert cov.complete_from == date(2026, 11, 23)


def test_drift_is_skipped_when_the_latest_day_is_small():
    rows = drift_rows(6)[:-30] + day_rows(PS5, "2026-11-22", [0] * 10 + [5] * 5 + [9] * 5)  # 20 reviews
    assert gold.meta(seeded_silver(rows), NOW)["drift"][PS5] is None


def test_drift_pools_the_prior_days_instead_of_averaging_shares():
    rows = day_rows(PS5, "2026-11-18", [2, 6, 9])
    rows += day_rows(PS5, "2026-11-19", [0] * 10 + [5] * 90) + day_rows(PS5, "2026-11-20", [0] * 10 + [5] * 90)
    rows += day_rows(PS5, "2026-11-21", [0] * 30)                       # tiny day, 100% low
    rows += day_rows(PS5, "2026-11-22", [0] * 50 + [5] * 50)            # latest, 50% low
    drift = gold.meta(seeded_silver(rows), NOW)["drift"][PS5]           # pooled prior share is 21.7%
    assert drift["metric"] == "low" and drift["delta_pp"] == pytest.approx(28.3, abs=0.1)


def test_all_view_sums_both_platforms_for_a_day():
    rows = day_rows(PS5, "2026-11-23", [9, 9]) + day_rows(XBOX, "2026-11-23", [1])
    (row,) = gold.daily(seeded_silver(rows), NOW)["all"]
    assert (row["n"], row["low"], row["high"]) == (3, 1, 2)
    assert row["mean"] == round(19 / 3, 2) and row["median"] == 9


def test_all_view_counts_a_text_repeated_across_platforms():
    rows = day_rows(PS5, "2026-11-23", [5, 5], ["copy", "own PS5 text"]) + day_rows(XBOX, "2026-11-23", [5], ["copy"])
    (row,) = gold.daily(seeded_silver(rows), NOW)["all"]
    assert row["repeated"] == round(2 / 3, 4)


def test_all_view_is_complete_only_when_every_platform_is():
    rows = (
        [(PS5, s, "2026-11-22", utc_ms(2026, 11, 22, 2), f"p{s}") for s in (9, 6, 2)]   # PS5 window starts the 22nd
        + [(XBOX, s, "2026-11-20", utc_ms(2026, 11, 20, 2), f"x{s}") for s in (9, 6, 2)]  # Xbox from the 20th
        + day_rows(XBOX, "2026-11-21", [5])
    )
    con = seeded_silver(rows)
    cov = gold.coverage(con)
    assert cov["all"].window_start == cov[PS5].window_start
    assert cov["all"].complete_from == cov[PS5].complete_from == date(2026, 11, 23)
    xbox_days = {r["date"]: r["complete"] for r in gold.daily(con, NOW)[XBOX]}
    all_days = {r["date"]: r["complete"] for r in gold.daily(con, NOW)["all"]}
    assert xbox_days["2026-11-21"] is True and all_days["2026-11-21"] is False


def test_all_view_hourly_adds_the_platforms_together():
    rows = [at("2026-11-23", 10, PS5, s) for s in (9, 6, 2)] + [at("2026-11-23", 10, XBOX, s) for s in (9, 6, 2)]
    hours = gold.hourly(seeded_silver(rows), NOW)
    assert hours["all"] == [{"hour": "2026-11-23T10:00:00Z", "n": 6}]


def test_all_view_without_any_rows_is_empty():
    con = seeded_silver([])
    assert gold.daily(con, NOW)["all"] == [] and gold.hourly(con, NOW)["all"] == []
    assert gold.coverage(con)["all"] == gold.Coverage(None, None)
