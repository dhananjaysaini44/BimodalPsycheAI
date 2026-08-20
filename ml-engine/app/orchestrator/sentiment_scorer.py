from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def score_turn_sentiment(text: str) -> float:
    """Fast, lightweight sentiment score for steering the conversation.
    NOT the final depression classifier — that runs once at session end."""
    return analyzer.polarity_scores(text)["compound"]  # -1 (negative) to +1 (positive)
