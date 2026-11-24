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
