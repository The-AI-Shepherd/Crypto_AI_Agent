# 🪙 AI Crypto Agent

This is the companion code for my YouTube video where we build a simplified AI Crypto Agent using Python and Flask.  
The agent can fetch crypto prices, generate charts, compare coins, and provide AI-driven analysis based on your queries.

## 🚀 Demo
Try it live: [https://aishepherd.dev/portfolio/crypto-agent](https://aishepherd.dev/portfolio/crypto-agent)

## 📹 Video
Watch the full video here: [https://www.youtube.com/watch?v=dT7XhyRpzNk](https://www.youtube.com/watch?v=dT7XhyRpzNk)

## 💡 Features
- Track historical crypto prices from CoinGecko API  
- Generate interactive price charts with Plotly  
- Compare multiple coins and visualize trends  
- Ask the AI agent questions about coins or trading signals  
- Get crypto news and updates  
- Session-based chart storage and temporary caching  
- Automatic background updates of crypto price data  

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/ai-crypto-agent.git
cd ai-crypto-agent
pip install -r requirements.txt
Create a .env file with the following:

SECRET_KEY=<your-secret-key>
OPENAI_API_KEY=<your-openai-api-key>          # https://platform.openai.com/account/api-keys
COIN_GECKO_API=<your-coingecko-api-key>      # https://www.coingecko.com/en/api
NEWS_API=<your-news-api-key>                  # https://newsapi.org/
DATABASE_URL='sqlite:///crypto_data.db'

Run the app:
python app.py

## 🧩 Usage
- Open the Flask app in your browser (http://127.0.0.1:5000)
- Navigate to Crypto Tracker to see price charts
- Navigate to AI Crypto Agent to ask questions and get AI-assisted analysis
- Use the Update Prices button to refresh historical data
- Charts are stored temporarily per session and cleared automatically

##⚡ Notes
- Requires OPENAI_API_KEY, COIN_GECKO_API, and NEWS_API to fetch AI responses, price data, and news
- The AI agent uses LangGraph under the hood to decide which tools to run based on user queries
- Temporary chart JSON files are stored in temp_charts and cleaned up automatically
