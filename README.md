# 📈 Apple Stock Price Predictor with News Sentiment 📰  
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)  
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)  
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FE4C02?style=for-the-badge&logo=streamlit)  
![Status](https://img.shields.io/badge/Status-Production-brightgreen?style=for-the-badge)

> 🚀 Real-time Apple stock predictor powered by deep learning, live news sentiment, and elegant visualizations.

🔗 **Live App**: [Click here to view](https://stockpredictor-djgqtzgnnyvgfnph5z2srn.streamlit.app/)

## 🔍 Overview

This app fetches **live AAPL stock prices**, scrapes **real-time news**, analyzes the **sentiment of top headlines**, and predicts the **next closing price** using a **dual-head LSTM model** trained on historical data.

You get a live dashboard with:
- 📈 Real-time chart
- 🧠 Sentiment score
- 📌 Price prediction
- 🔺 Direction forecast
- ⏱️ Timestamp with **auto-refresh** & **manual refresh button**

## 🚀 Features

✅ **Live Stock Data** from `yfinance`  
📰 **News Scraping** using `NewsAPI`  
💬 **Sentiment Analysis** with `TextBlob`  
🧠 **LSTM Neural Network** (PyTorch)  
📊 **Predicted Price & Movement Direction**  
📉 **7-Day Trend Visualization**  
🔁 **Auto-refresh and Manual Refresh Button**

## 🧠 Model Architecture

Model: DualHeadLSTM  
├── LSTM (2 layers, 64 hidden units)  
├── Regressor head → Predicts closing price  
└── Classifier head → Predicts direction (UP/DOWN)  

- **Inputs**: `[Open, Close, Sentiment]`  
- **Outputs**:  
  - `price`: predicted close value  
  - `direction`: probability of price rising  

## 📂 Folder Structure

📁 Stock_Predictor/  
├── model.py          # Model architecture (DualHeadLSTM)  
├── utils.py          # Utilities: news fetch, sentiment, preprocessing  
├── streamlit.py      # Streamlit dashboard  
├── stock_model.pt    # Trained PyTorch model  
├── requirements.txt  # Dependencies  
└── README.md         # Project description  

## ⚙️ Setup Instructions

# Clone the Repo  
git clone https://github.com/your-username/stock-predictor.git  
cd stock-predictor  

# Install Requirements  
pip install -r requirements.txt  

# Run Streamlit App  
streamlit run streamlit.py  

## 🔑 API Key

- You'll need a free NewsAPI key from https://newsapi.org  
- Paste it inside `utils.py` in the `fetch_apple_news()` function.  

## 📌 Example Output

📊 Prediction Engine  

Open Price     →  $195.23  
Close Price    →  $196.42  
Sentiment      →  +0.24  
Predicted Close →  $197.90  
Direction       →  🔺 UP  

## 🙌 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss.

## 📄 License

This project is licensed under the MIT License.
