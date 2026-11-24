from datetime import UTC, date, datetime

from helpers import seeded_silver, utc_ms

from reviews_pipeline import gold

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
PS5, XBOX = "playstation-5", "xbox-series-x"


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


def test_valid_days_allows_one_day_ahead():
    assert gold.valid_days(NOW) == date(2026, 11, 25)


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
        + day_rows(PS5, "2026-11-25", [5])   # one day ahead: kept
        + day_rows(PS5, "2026-11-26", [5])   # two days ahead: junk
        + day_rows(PS5, "2027-11-21", [5])
    )
    days = [r["date"] for r in only(gold.daily(seeded_silver(rows), NOW))]
    assert days == ["2026-11-23", "2026-11-25"]


def test_daily_platform_without_rows_is_an_empty_list():
    out = gold.daily(seeded_silver(day_rows(PS5, "2026-11-23", [5])), NOW)
    assert out[XBOX] == []
    assert set(out) == {PS5, XBOX}


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
    assert meta["drift"] == {PS5: None, XBOX: None}


def drift_rows(latest_low):
    """Three earlier days with 10% low scores, then a latest day with `latest_low` low scores."""
    steady = [0] + [5] * 8 + [9]
    rows = day_rows(PS5, "2026-11-18", [2, 6, 9])  # opens every band, so 19th onwards is complete
    for day in ("2026-11-19", "2026-11-20", "2026-11-21"):
        rows += day_rows(PS5, day, steady)
    return rows + day_rows(PS5, "2026-11-22", [0] * latest_low + [5] * (9 - latest_low) + [9])


def test_drift_flags_a_large_move_in_a_band_share():
    drift = gold.meta(seeded_silver(drift_rows(6)), NOW)["drift"][PS5]
    assert drift == {"date": "2026-11-22", "metric": "low", "delta_pp": 50.0}


def test_drift_ignores_a_move_within_the_threshold():
    assert gold.meta(seeded_silver(drift_rows(2)), NOW)["drift"][PS5] is None


def test_drift_is_null_with_fewer_than_three_prior_days():
    rows = day_rows(PS5, "2026-11-18", [2, 6, 9]) + day_rows(PS5, "2026-11-19", [0] * 9 + [9])
    assert gold.meta(seeded_silver(rows), NOW)["drift"][PS5] is None


def test_build_has_the_three_documents():
    out = gold.build(seeded_silver(day_rows(PS5, "2026-11-23", [5])), NOW)
    assert set(out) == {"meta", "daily", "hourly"}
    assert set(out["daily"]["platforms"]) == {PS5, XBOX}
    assert set(out["hourly"]["platforms"]) == {PS5, XBOX}
