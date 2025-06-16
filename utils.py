import requests
import pandas as pd
from textblob import TextBlob

def fetch_apple_news(api_key= "10c053c8a02e41b9b6cfc0e63e75578c", query="Apple Inc", num_articles=5):
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"
    res = requests.get(url)
    articles = res.json().get("articles", [])[:num_articles]
    return articles

def analyze_sentiment(headlines):
    sentiments = []
    for h in headlines:
        blob = TextBlob(h)
        sentiments.append(blob.sentiment.polarity)
    return sum(sentiments) / len(sentiments) if sentiments else 0
