import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"
DEFAULT_TICKER = "BTC-USD"
DEFAULT_PERIOD = "1mo"
CHART_FILENAME = "crypto_sentiment_chart.png"
SENTIMENT_MAP = {"positive": 1, "negative": -1, "neutral": 0}


def load_sentiment_model(model_name: str = "ProsusAI/finbert"):
    """Load the FinBERT sentiment analysis pipeline."""
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def fetch_news(news_api_key: str, query: str = "bitcoin", language: str = "en") -> pd.DataFrame:
    """Fetch Bitcoin news from NewsAPI and return a cleaned DataFrame."""
    response = requests.get(
        NEWS_API_ENDPOINT,
        params={"q": query, "language": language, "apiKey": news_api_key},
        timeout=15,
    )
    response.raise_for_status()

    articles = response.json().get("articles", [])
    news_data = [
        {
            "Date": pd.to_datetime(article.get("publishedAt", ""), errors="coerce")
            .normalize()
            .tz_localize(None),
            "Text": ". ".join(
                text for text in (article.get("title"), article.get("description"))
                if text
            ),
        }
        for article in articles
    ]

    return pd.DataFrame(news_data).dropna(subset=["Date", "Text"]).reset_index(drop=True)


def fetch_price_history(ticker: str = DEFAULT_TICKER, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Fetch recent ticker history from Yahoo Finance."""
    history = yf.Ticker(ticker).history(period=period).reset_index()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None).dt.normalize()
    return history[["Date", "Close"]]


def compute_daily_sentiment(news_df: pd.DataFrame, analyzer):
    """Run sentiment scoring and aggregate scores by date."""
    print("Running sentiment analysis...")

    def get_score(text: str) -> int:
        try:
            label = analyzer(text, truncation=True, max_length=512)[0]["label"].lower()
            return SENTIMENT_MAP.get(label, 0)
        except Exception:
            return 0

    news_df = news_df.copy()
    news_df["Score"] = news_df["Text"].apply(get_score)
    return news_df.groupby("Date", as_index=False)["Score"].mean()


def create_chart(final_df: pd.DataFrame, output_path: str) -> None:
    """Draw and save the sentiment vs BTC price correlation chart."""
    print("Generating chart...")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sentiment Score", color="blue")
    ax1.bar(final_df["Date"], final_df["Score"], color="blue", alpha=0.3, label="Sentiment")

    ax2 = ax1.twinx()
    ax2.set_ylabel("BTC Price (USD)", color="orange")
    ax2.plot(final_df["Date"], final_df["Close"], color="orange", marker="o", linewidth=2, label="Price")

    plt.title("Bitcoin Sentiment (FinBERT) vs Price Correlation")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Bitcoin sentiment and price chart.")
    parser.add_argument(
        "--output",
        default=CHART_FILENAME,
        help=f"Output chart filename (default: {CHART_FILENAME})",
    )
    parser.add_argument(
        "--skip-chart",
        action="store_true",
        help="Skip chart creation and only fetch and process data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    news_api_key = os.getenv("NEWS_API_KEY", "9c3fa5263e7e458b976de44d657d0add")
    if not news_api_key:
        raise ValueError("Missing NewsAPI key: set NEWS_API_KEY environment variable.")

    sentiment_pipe = load_sentiment_model()
    news_df = fetch_news(news_api_key)
    price_df = fetch_price_history()

    if news_df.empty or price_df.empty:
        print("Could not find sufficient data. Check your API key and internet connection.")
        return

    daily_sentiment = compute_daily_sentiment(news_df, sentiment_pipe)
    final_df = pd.merge(daily_sentiment, price_df, on="Date", how="inner").sort_values("Date")

    if final_df.empty:
        print("No matching dates found to create a chart.")
        return

    if args.skip_chart:
        print("Chart creation skipped. Data processing completed successfully.")
        return

    create_chart(final_df, os.path.join(os.getcwd(), args.output))


if __name__ == "__main__":
    main()