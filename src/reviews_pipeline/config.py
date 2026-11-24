"""Constants for the Metacritic collection."""

BASE_URL = (
    "https://backend.metacritic.com/reviews/metacritic/user/games/"
    "grand-theft-auto-vi/platform/{platform}/web"
)
PLATFORMS = ("playstation-5", "xbox-series-x")
SENTIMENTS = ("positive", "neutral", "negative")
PAGE_LIMIT = 100
MAX_OFFSET = 9900  # each query reaches 10,000 results; offset 10000 returns HTTP 500
MIN_INTERVAL_S = 3.0
USER_AGENT = "reviews-gta6-pipeline (portfolio project; polite scheduled collector)"
