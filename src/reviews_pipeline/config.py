"""Constants for the Metacritic collection."""

BASE_URL = (
    "https://backend.metacritic.com/reviews/metacritic/user/games/"
    "grand-theft-auto-vi/platform/{platform}/web"
)
# Platform registry: the slug used in the endpoint URL and the name shown on the site.
# Adding a platform here is the only change needed to collect it, aggregate it and list it
# (and include it in "All platforms") on the site.
PLATFORM_LABELS = {"playstation-5": "PlayStation 5", "xbox-series-x": "Xbox Series X"}
PLATFORMS = tuple(PLATFORM_LABELS)
SENTIMENTS = ("positive", "neutral", "negative")
PAGE_LIMIT = 200  # verified live: limit=200 returns 200 items
MAX_OFFSET = 9800  # last full page starts here; each query reaches 10,000 results and offset 10000 returns HTTP 500
MIN_INTERVAL_S = 3.0
USER_AGENT = "reviews-gta6-pipeline (portfolio project; polite scheduled collector)"
