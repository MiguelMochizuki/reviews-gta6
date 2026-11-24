"""Constants for the Metacritic collection."""

BASE_URL = (
    "https://backend.metacritic.com/reviews/metacritic/user/games/"
    "grand-theft-auto-vi/platform/{platform}/web"
)
PLATFORMS = ("playstation-5", "xbox-series-x")
SENTIMENTS = ("positive", "neutral", "negative")
PAGE_LIMIT = 200  # verified live: limit=200 returns 200 items
MAX_OFFSET = 9800  # last full page starts here; each query reaches 10,000 results and offset 10000 returns HTTP 500
MIN_INTERVAL_S = 3.0
USER_AGENT = "reviews-gta6-pipeline (portfolio project; polite scheduled collector)"
