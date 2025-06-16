import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf

from model import DualHeadLSTM
from utils import fetch_apple_news, analyze_sentiment

# ------------------- Auto-refresh ------------------- #
st.set_page_config(layout="centered")
st_autorefresh(interval=60 * 1000, key="refresh")

# ------------------- Header ------------------- #
st.title("📈 Apple Stock Price Predictor with News Sentiment")

# ------------------- Manual Refresh ------------------- #
if st.button("🔁 Manual Refresh"):
    st.rerun()

# ------------------- Load Model ------------------- #
model = DualHeadLSTM()
model.load_state_dict(torch.load("stock_model.pt", map_location=torch.device("cpu")))
model.eval()

# ------------------- Fetch Stock Data ------------------- #
ticker = "AAPL"
stock_data = yf.download(ticker, period="7d", interval="1h")[['Open', 'Close']]
stock_data.dropna(inplace=True)
stock_data.reset_index(inplace=True)

# Display last update time
st.caption(f"⏱️ Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------- News and Sentiment ------------------- #
articles = fetch_apple_news()
headlines = [a['title'] for a in articles]
sentiment_score = analyze_sentiment(headlines)

st.subheader("📰 Latest News & Sentiment")
for headline in headlines:
    st.markdown(f"• {headline}")
st.write(f"**Sentiment Score:** {round(sentiment_score, 3)}")

# ------------------- Prediction ------------------- #
st.subheader("📊 Prediction Engine")

# Prepare input sequence
latest_data = stock_data[['Open', 'Close']].values[-10:]
if len(latest_data) < 10:
    st.error("Not enough data to make prediction.")
    st.stop()

# Append sentiment to each timestep
sent_seq = np.full((10, 1), sentiment_score)
input_seq = np.hstack((latest_data, sent_seq))
input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    pred_price, pred_direction = model(input_tensor)
    pred_price = pred_price.item()
    pred_direction = pred_direction.item()

# Display values
latest_open = stock_data['Open'].iloc[-1]
latest_close = stock_data['Close'].iloc[-1]
delta = pred_price - latest_close

col1, col2, col3 = st.columns(3)
col1.metric("Open Price", f"${latest_open.item():.2f}")
col2.metric("Close Price", f"${latest_close.item():.2f}")
col3.metric("📌 Predicted Close", f"${pred_price:.2f}", delta=f"{delta.item():+.2f}")

# Direction
arrow = "🔺 UP" if pred_direction > 0.5 else "🔻 DOWN"
st.markdown(f"**Direction:** {arrow}")

# ------------------- Live Price Graph ------------------- #
st.subheader("📉 Apple Stock Trend (Past 7 Days)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(stock_data['Datetime'], stock_data['Close'], label='Close Price', color='blue')
ax.plot(stock_data['Datetime'], stock_data['Open'], label='Open Price', color='orange')
ax.set_title("AAPL Stock Prices")
ax.set_xlabel("Datetime")
ax.set_ylabel("Price (USD)")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
