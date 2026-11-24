import re
from datetime import UTC, datetime
from pathlib import Path

from helpers import seeded_silver, utc_ms

from reviews_pipeline import gold

SITE = Path(__file__).parent.parent / "site"
NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
ALLOWED_URLS = ("https://www.metacritic.com", "https://github.com/MiguelMochizuki/reviews-gta6")


def built():
    rows = []
    for day in (21, 22, 23):  # three bands open on the 21st, so later days are complete
        for hour, score in ((10, 2), (11, 6), (12, 9)):
            rows.append(("playstation-5", score, f"2026-11-{day}", utc_ms(2026, 11, day, hour),
                         f"t{day}{hour}"))
    return gold.build(seeded_silver(rows), NOW)


def declared_fields():
    """Names the page reads, from the `// fields:` lines at the top of app.js."""
    source = (SITE / "app.js").read_text()
    names = {}
    for line in re.findall(r"^// fields:(.*)$", source, flags=re.MULTILINE):
        for token in line.split():
            kind, _, key = token.partition(".")
            names.setdefault(kind, set()).add(key)
    return names


def test_app_reads_only_keys_that_gold_writes():
    out, names = built(), declared_fields()
    platform = out["daily"]["platforms"]["playstation-5"]
    hour_rows = out["hourly"]["platforms"]["playstation-5"]
    assert platform and hour_rows  # the fixture must exercise every shape
    assert names["meta"] and names["row"] and names["hour"]
    assert names["meta"] <= set(out["meta"])
    assert names["row"] <= set(platform[0])
    assert names["hour"] <= set(hour_rows[0])


def test_page_loads_nothing_from_the_network_except_its_own_files():
    for name in ("index.html", "app.js", "style.css"):
        text = (SITE / name).read_text()
        for url in re.findall(r"https?://[^\s\"')<>]+", text):
            assert url.startswith(ALLOWED_URLS), (name, url)
    html = (SITE / "index.html").read_text()
    assert 'src="vendor/chart.umd.min.js"' in html and 'src="app.js"' in html


def test_page_names_the_source_and_says_aggregates_only():
    html = (SITE / "index.html").read_text()
    assert "Metacritic user reviews" in html
    assert "aggregates only" in html


def test_vendored_chart_library_keeps_its_license():
    assert "MIT" in (SITE / "vendor" / "CHART_LICENSE.txt").read_text()
