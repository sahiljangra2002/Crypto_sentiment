# 📈 Bitcoin Sentiment & Price Correlation Model

An AI-driven tool that analyzes live Bitcoin news headlines using **FinBERT** and correlates them with real-time price data from **Yahoo Finance**.

## 🚀 Overview
This project explores the relationship between market sentiment and cryptocurrency price movements. It uses a specialized NLP model, **FinBERT**, which is pre-trained on financial communication to provide more accurate sentiment scoring than standard models.

## 📊 Results
![Bitcoin Correlation Chart](sentiment_chart.png)
*Above: The dual-axis chart showing daily average sentiment vs. BTC closing price.*

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **NLP Model:** `ProsusAI/finbert` (via HuggingFace Transformers)
- **Data Sources:** NewsAPI (Headlines), yfinance (Market Data)
- **Visualization:** Matplotlib

## 📖 GATE DA Connection
This project applies several core concepts from the **Data Science & AI (DA)** syllabus:
- **Probability & Statistics:** Correlation analysis between two variables.
- **Machine Learning:** Utilizing pre-trained Transformers for classification tasks.
- **Data Wrangling:** Merging time-series data from multiple APIs.

