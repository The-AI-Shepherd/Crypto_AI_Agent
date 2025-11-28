# 1️⃣ Standard Library Imports
import datetime
import datetime as dt
import operator
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Annotated, Dict, List, TypedDict, Union

# 2️⃣ Third-Party Library Imports
import numpy as np
import pandas as pd
import plotly.io as pio
import requests
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.caches import BaseCache
from langchain_core.callbacks.base import Callbacks
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from plotly.graph_objs import Figure, Scatter
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.stats import linregress
from itertools import combinations

# 3️⃣ Local Application Imports
from app_factory import create_app, logging
from database_models import CryptoMarketData, db
from enum_classes import CryptoCoins, CryptoValues
from llm_prompting import (
    Bollinger_Bands_interpretation_prompt,
    EMA_interpretation_prompt,
    MACD_interpretation_prompt,
    OBV_interpretation_prompt,
    RSI_interpretation_prompt,
    SMA_interpretation_prompt,
    result_synthesis_prompt,
    langgraph_faq_prompt,
    news_synthesis_prompt,
    tool_selection_prompt
)

load_dotenv()
ChatOpenAI.model_rebuild()


class AgentState(TypedDict):
    final_prompt: Annotated[str, "llm final prompt"]
    tool_prompt: Annotated[str, "llm tool selection prompt"]
    messages: Annotated[Union[dict, List[dict]], operator.add]
    used_tools: Annotated[List[dict], operator.add]


def track_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_report = {
                "success": False,
                "error": str(e),
                "function": func.__name__
            }
            logging.exception(error_report)
            return error_report
    return wrapper


class CryptoAgent:
    def __init__(self):
        self.debugging = False
        self.crypto_chart_list = []
        self.max_coins = 4

        # Values added to prompts
        self.session_coin = "bitcoin"
        self.user_query = ""
        self.date_today = dt.datetime.now(datetime.UTC)

        # Tool list set up
        self.tool_string_list = ["performance_tools", "performance_tools", "comparison_tools", "portfolio_tools",
                                 "fundamental_tools", "other_tools"]
        self.performance_tools = [_wrapped_get_crypto_price_range, _get_basic_crypto_price_chart, _get_volume_chart,
                                  _get_capitalization_chart]
        self.comparison_tools = [_compare_cryptos]
        self.portfolio_tools = [_get_macd_chart, _get_obv_chart, _get_rsi_chart, _get_ema_chart, _get_sma_chart,
                                _get_bollinger_bands_chart]
        self.fundamental_tools = [_wrapped_get_crypto_news, _crypto_faq]
        self.other_tools = [_crypto_faq, _wrapped_get_crypto_news]

        self.tools = (self.performance_tools + self.comparison_tools + self.portfolio_tools + self.fundamental_tools +
                      self.other_tools)

        # Supported coin list
        self.supported_coin_list = "\n".join([f"{coin.value["name"]}: "
                                              f"[{coin.value["name"].title()},"
                                              f"{coin.value["symbol"].upper()}, "
                                              f"{coin.value["id"]}, "
                                              f"{coin.value["symbol"].lower()}]" for coin in CryptoCoins])

        # Initialize LLM and agent globally
        self.llm_gpt_o4 = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=1)
        self.llm_gpt5 = init_chat_model(model="gpt-5-mini", model_provider="openai", temperature=1)
        self.llm_with_tools = self.llm_gpt_o4.bind_tools(tools=self.tools)

        # Initialize graph
        self.graph = self._build_graph()

        # Initialize app to query db
        self.app = create_app()

    def set_session_coin(self, session_coin: list) -> None:
        """
        Sets the values for the session_coin attribute in the CryptoAgent class
        Parameters
        ----------
        session_coin : list
            crypto coin name(s), (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
        """
        self.session_coin = session_coin

    def get_chart_list(self):
        """
        returns a list of dicts with generated charts
        Returns
        -------
        list of dicts with following keys
        "chart_id": "basic",
        "chart_name": "Basic price chart",
        "chart_html": chart_html,
        "plotly_fig": fig,
        "coin": crypto_name
        """
        return self.crypto_chart_list

    def get_coin_list(self, state):
        """
        Returns of list of coin name(s) found in user query
        Returns
        -------
        list
            a list of strings
        """
        coin_list_with_dicts = state.get("coins", [])
        if coin_list_with_dicts:
            coin_list = [coin["coin_name"] for coin in coin_list_with_dicts]
            return coin_list
        else:
            return []

    @track_errors
    def get_crypto_price_range(self, crypto_name: str, min_date: str, max_date: str) -> dict:
        """
        Fetches price data of a cryptocurrency in USD over a date range. Also used for general price
        analysis or trends for a supported coin over a period.
        Parameters
        ----------
        crypto_name : str
            crypto_name: Name of the cryptocurrency, (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".

        Returns
        -------
        dict
            a normalized dict containing price data information about a specific coin from a date range
            -"crypto coin": str,
            -"crypto symbol": str,
            -"image url": str,
            -"price data": list[float],
            -"market capitalization": list[float],
            -"total volume": list[float],
            -"dates": list[DateTime]
            -"price summary": dict
        """
        try:
            min_date_to_fetch = dt.datetime.strptime(
                min_date.replace('"', '').replace("'", "").strip(), "%Y-%m-%d")
            max_date_to_fetch = dt.datetime.strptime(
                max_date.replace('"', '').replace("'", "").strip(), "%Y-%m-%d")

            with self.app.app_context():
                crypto_price_range_raw = db.session.query(CryptoMarketData).filter(
                    CryptoMarketData.name == crypto_name.title(),
                    CryptoMarketData.date >= min_date_to_fetch,
                    CryptoMarketData.date <= max_date_to_fetch
                ).order_by(CryptoMarketData.date.desc())

                results = crypto_price_range_raw.all()

                if results:
                    price_list = []
                    market_cap_list = []
                    total_volume_list = []
                    date_list = []

                    for price_data in results:
                        price_list.append(price_data.current_price)
                        market_cap_list.append(price_data.market_cap)
                        total_volume_list.append(price_data.total_volume)
                        date_list.append(price_data.date)

                    start_price = price_list[-1]  # The list of price data is descending from newest to oldest
                    end_price = price_list[0]

                    # Summary calculations
                    price_summary = {
                        "min_price": min(price_list),
                        "max_price": max(price_list),
                        "average_price": round(sum(price_list) / len(price_list), 2),
                        "start_price": start_price,
                        "end_price": end_price,
                        "price_change": round(end_price - start_price, 2),
                        "percentage_change": round(((end_price - start_price) / start_price) * 100, 2)
                        if price_list[0] != 0 else None
                    }

                    sample = results[0]

                    crypto_price_range = {
                        "crypto coin": sample.name,
                        "crypto symbol": sample.symbol,
                        "image url": sample.image,
                        "price data": price_list,
                        "market capitalization": market_cap_list,
                        "total volume": total_volume_list,
                        "dates": date_list,
                        "start date": min(date_list),
                        "end date": max(date_list),
                        "price summary": price_summary
                    }

                    return crypto_price_range

                else:
                    return {"error": f"I couldn't find any price data for {crypto_name} between {min_date} "
                                     f"and {max_date}."}

        except Exception as e:
            error_message = f"Error occurred while fetching range price data: {e}"
            raise error_message

    @track_errors
    def get_crypto_price_specific(self, crypto_name: str, date: str) -> dict:
        """
        Fetches price data of a cryptocurrency in USD from a specific date.

        Parameters
        ----------
        crypto_name : str
            crypto_name: Name of the cryptocurrency, (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
        date : str
            date in "YYYY-MM-DD".


        Returns
        -------
        dict
            a normalized dict containing price data information about a specific coin on a specific date
            -"crypto coin": str,
            -"crypto symbol": str,
            -"image url": str,
            -"price data": float,
            -"market capitalization": float,
            -"total volume": float,
            -"date": DateTime object
        """

        try:
            with self.app.app_context():
                date_to_fetch = dt.datetime.strptime(date, "%Y-%m-%d")
                crypto_price_specific_raw = db.session.query(
                    CryptoMarketData).filter(CryptoMarketData.name == crypto_name.title(),
                                             CryptoMarketData.date == date_to_fetch)

                if crypto_price_specific_raw:
                    crypto_price_data_single = crypto_price_specific_raw.first()

                    crypto_price_specific = {
                        "crypto coin": crypto_price_data_single.name,
                        "crypto symbol": crypto_price_data_single.symbol,
                        "image url": crypto_price_data_single.image,
                        "price data": crypto_price_data_single.current_price,
                        "market capitalization": crypto_price_data_single.market_cap,
                        "total volume": crypto_price_data_single.total_volume,
                        "date": crypto_price_data_single.date
                    }

                    return crypto_price_specific

                else:
                    return {"error": f"I Couldn't find any price data for {crypto_name} on {date}."}

        except Exception as e:
            error_message = f"Error occurred while fetching specific day price data: {e}"
            raise error_message

    @track_errors
    def get_crypto_news(self, crypto_name: str, min_date: str, max_date: str = None) -> dict:
        """
        Fetches news articles about a given cryptocurrency from a given date range.

        Parameters
        ----------
        crypto_name : str
            crypto_name: Name of the cryptocurrency, (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD". (optional)

        Returns
        -------
        dict
            a dict containing a list of articles found, the minimum search date and maximum search date
            -"articles": list,
            -"min_date": str
            -"max_date": str
        """
        news_api_key = os.getenv('NEWS_API')
        headers = {"x-api-Key": news_api_key}

        if max_date:
            logging.info(f"Fetching relevant {crypto_name} news between {min_date} and {max_date}...")
            news_parameters = {"q": f"crypto AND {crypto_name}",
                               "from": min_date,
                               "to": max_date
                               }
            news_date = f"{min_date} - {max_date}"

        else:
            logging.info(f"Fetching relevant {crypto_name} news on {min_date}...")
            news_parameters = {"q": crypto_name,
                               "from": min_date,
                               "sortBy": "popularity",
                               }
            news_date = min_date

        # Using the News API endpoint, we fetch the news from the indicated news date:
        response = requests.get("https://newsapi.org/v2/everything?", params=news_parameters, headers=headers)

        if "application/json" in response.headers["Content-Type"]:
            news_data = response.json()
            logging.info(f"Found {news_data.get("totalResults")} results.")
            articles = []
            number_of_articles_to_use = 10
            try:
                # Use a limited number of articles to avoid article clutter
                if int(news_data.get("totalResults")) > number_of_articles_to_use:
                    number_of_articles_to_read = number_of_articles_to_use
                else:
                    number_of_articles_to_read = int(news_data.get("totalResults"))

                for article in news_data["articles"][:number_of_articles_to_read]:
                    article_info = {
                        "author": article.get("author"),
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "source": article.get("url"),
                        "published_at": article.get("publishedAt"),
                        "content": article.get("content")
                    }
                    articles.append(article_info)

                return {"news_articles": articles, "min_date": min_date, "max_date": max_date}
            except Exception as e:
                error_message = f"There was an error getting news articles. Error: {e}"
                raise error_message
        else:
            error_message = (f"There was an issue in getting news data, invalid JSON received from "
                             f"the news source, response status code: {response.status_code}")
            raise error_message

    def _make_price_df(self, price_data):
        """
        Create a pandas DataFrame from the normalized price_data dict.

        Parameters
        ----------
        price_data : dict
            Normalized price data as returned by `_ensure_price_data`.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by datetime (date) with the following columns:
            - "price" (float)
            - "volume" (float or NaN)
            - "cap" (float or NaN)

        Notes
        -----
        - The function ensures the dates are parsed to pandas.Timestamp and sorted ascending.
        - Missing volume/cap values will be filled with NaN.
        """
        df = pd.DataFrame({
            "date": pd.to_datetime(price_data["dates"]),
            "price": price_data["price data"],
            "volume": price_data.get("total volume", [None] * len(price_data["dates"])),
            "cap": price_data.get("market capitalization", [None] * len(price_data["dates"]))
        })
        df = df.sort_values("date", ascending=False).set_index("date")
        return df

    @track_errors
    def get_basic_crypto_price_chart(self, crypto_name: str, min_date: str, max_date: str) -> dict:
        """
        Creates a basic crypto price chart from the min_date to the max_date for the given crypto_name

        Parameters
        ----------
        crypto_name : str
            crypto_name: Name of the cryptocurrency, (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".

        Returns
        -------
        dict
        """
        try:
            price_data = self.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)

            if "error" in price_data:
                return price_data

            df = self._make_price_df(price_data)

            dates = df.index.strftime("%Y-%m-%d").tolist()
            prices = df["price"]

            fig = Figure()
            fig.add_trace(Scatter(x=dates, y=prices, mode='lines+markers', name='Price'))
            fig.update_layout(
                title=f'{crypto_name.upper()} Price Over Time',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                template='plotly_white',
                height=400
            )

            chart_html = pio.to_html(fig, full_html=False)

            self.crypto_chart_list.append({"chart_id": "basic",
                                           "chart_name": f"{crypto_name.title()} price chart",
                                           "chart_html": chart_html,
                                           "plotly_fig": fig,
                                           "coin": crypto_name})

            logging.info(f"Created a basic chart for {crypto_name} between {min_date} and {max_date}")

            return {
                "indicator": "price_only",
                "params": {"min_date": min_date, "max_date": max_date},
                "price_summary": price_data.get("price summary"),
            }

        except Exception as e:
            error_message = f"Error creating chart: {e}"
            raise error_message

    @track_errors
    def get_capitalization_chart(self, crypto_name: str, min_date: str, max_date: str) -> dict:
        """
        Creates a crypto market capitalization chart from the min_date to the max_date for the given crypto_name

        Parameters
        ----------
        crypto_name : str
            crypto_name: Name of the cryptocurrency, (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".

        Returns
        -------
        dict
        """
        try:
            price_data = self.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)

            if "error" in price_data:
                return price_data

            df = self._make_price_df(price_data)

            dates = df.index.strftime("%Y-%m-%d").tolist()
            market_capitalization = df["cap"]

            # Summary calculations
            capitalization_summary = {
                "min_capitalization": min(market_capitalization),
                "max_capitalization": max(market_capitalization),
                "average_capitalization": round(sum(market_capitalization) / len(market_capitalization), 2),
                "start_capitalization": market_capitalization.iloc[0],
                "end_capitalization": market_capitalization.iloc[-1],
                "capitalization_change": round(market_capitalization.iloc[-1] - market_capitalization.iloc[0], 2),
                "percentage_change": round(
                    ((market_capitalization.iloc[-1] - market_capitalization.iloc[0]) / market_capitalization.iloc[0]) * 100, 2)
                if market_capitalization.iloc[0] != 0 else None
            }

            fig = Figure()
            fig.add_trace(Scatter(x=dates, y=market_capitalization, mode='lines+markers',
                                  name='Market Capitalization'))
            fig.update_layout(
                title=f'{crypto_name.upper()} Price Over Time',
                xaxis_title='Date',
                yaxis_title='Market Capitalization (USD)',
                template='plotly_white',
                height=400
            )

            chart_html = pio.to_html(fig, full_html=False)

            self.crypto_chart_list.append({"chart_id": "capitalization",
                                           "chart_name": "Market capitalization chart",
                                           "chart_html": chart_html,
                                           "plotly_fig": fig,
                                           "coin": crypto_name})

            logging.info(f"Created a market capitalization chart for {crypto_name} between {min_date} and {max_date}")

            return {
                "indicator": "market_capitalization",
                "params": {"min_date": min_date, "max_date": max_date},
                "price_summary": price_data.get("price summary"),
                "capitalization_summary": capitalization_summary,
            }

        except Exception as e:
            error_message = f"Error creating chart: {e}"
            raise error_message

    @track_errors
    def get_volume_chart(self, crypto_name: str, min_date: str, max_date: str) -> dict:
        """
        Creates a total volume chart from the min_date to the max_date for the given crypto_name

        Parameters
        ----------
        crypto_name : str
            crypto_name: Name of the cryptocurrency, (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".

        Returns
        -------
        dict
        """
        try:
            price_data = self.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)

            if "error" in price_data:
                return price_data

            df = self._make_price_df(price_data)

            dates = df.index.strftime("%Y-%m-%d").tolist()
            volumes = df["volume"]

            # Summary calculations
            volume_summary = {
                "min_capitalization": min(volumes),
                "max_capitalization": max(volumes),
                "average_capitalization": round(sum(volumes) / len(volumes), 2),
                "start_capitalization": volumes.iloc[0],
                "end_capitalization": volumes.iloc[-1],
                "capitalization_change": round(volumes.iloc[-1] - volumes.iloc[0], 2),
                "percentage_change": round(
                    ((volumes.iloc[-1] - volumes.iloc[0]) / volumes.iloc[0]) * 100, 2)
                if volumes.iloc[0] != 0 else None
            }

            fig = Figure()
            fig.add_trace(Scatter(x=dates, y=volumes, mode='lines+markers', name='Total volume'))
            fig.update_layout(
                title=f'{crypto_name.upper()} Market Volume Over Time',
                xaxis_title='Date',
                yaxis_title='Total Volume (USD)',
                template='plotly_white',
                height=400
            )

            chart_html = pio.to_html(fig, full_html=False)

            self.crypto_chart_list.append({"chart_id": "volume",
                                           "chart_name": "Total volume chart",
                                           "chart_html": chart_html,
                                           "plotly_fig": fig,
                                           "coin": crypto_name})

            logging.info(f"Created a volume chart for {crypto_name} between {min_date} and {max_date}")

            return {
                "indicator": "total_volume",
                "params": {"min_date": min_date, "max_date": max_date},
                "price_summary": price_data.get("price summary"),
                "volume_summary": volume_summary,
            }

        except Exception as e:
            error_message = f"Error creating chart: {e}"
            raise error_message

    @track_errors
    def get_sma_chart(self, crypto_name: str, min_date: str, max_date: str, window: int = 20) -> dict:
        """
        Generate a price chart with a Simple Moving Average (SMA).
        The SMA smooths out price data over a chosen period.
        Interpretation:
         - Helps identify overall trend direction.
         - Price above SMA may indicate uptrend; below SMA may indicate downtrend.
         - Longer SMAs (e.g., 200-day) act as support/resistance in many trading strategies.

        Parameters
        ----------
        crypto_name : str
            Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".
        window : int, optional
            Rolling window length for SMA. Default is 20.

        Returns
        -------
        dict
            JSON-serializable dict containing:
            - "chart_html": str (Plotly HTML fragment, safe to render via Jinja)
            - "indicator": "SMA"
            - "params": dict (e.g., {"window": 20})
            - "price_summary": dict (from the price fetcher) or None

        Error handling
        --------------
        Returns {"error": "message"} on failure.
        """
        try:
            price_data = self.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)
            if "error" in price_data:
                return price_data

            df = self._make_price_df(price_data)
            df[f"SMA_{window}"] = df["price"].rolling(window=window, min_periods=1).mean()

            dates = df.index.strftime("%Y-%m-%d").tolist()

            fig = Figure()
            fig.add_trace(Scatter(x=dates, y=df["price"], mode="lines", name="Price"))
            fig.add_trace(Scatter(x=dates, y=df[f"SMA_{window}"], mode="lines", name=f"SMA {window}"))
            fig.update_layout(title=f"{crypto_name.upper()} Price + SMA{window}", xaxis_title="Date",
                              yaxis_title="Price (USD)", template="plotly_white", height=420)

            sma_col = f"SMA_{window}"
            sma = df[sma_col]
            price = df["price"]

            last_sma = sma.iloc[-1]
            last_price = price.iloc[-1]

            # Price position relative to SMA
            price_above_sma = last_price > last_sma

            # SMA trend slope over last 'window' days
            recent_sma = sma.iloc[-window:]
            slope, _, _, _, _ = linregress(np.arange(len(recent_sma)), recent_sma)

            # Percentage difference between price and SMA
            perc_diff = ((last_price - last_sma) / last_sma) * 100 if last_sma != 0 else 0

            # Recent crossings (simplified example over last window)
            price_series = price.iloc[-window:]
            sma_series = sma.iloc[-window:]
            cross_ups = ((price_series > sma_series) & (price_series.shift(1) <= sma_series.shift(1))).sum()
            cross_downs = ((price_series < sma_series) & (price_series.shift(1) >= sma_series.shift(1))).sum()

            summary = {
                "last_sma": last_sma,
                "last_price": last_price,
                "price_above_sma": price_above_sma,
                "sma_trend_slope": slope,
                "price_sma_perc_diff": perc_diff,
                "recent_cross_ups": int(cross_ups),
                "recent_cross_downs": int(cross_downs)
            }

            chart_html = pio.to_html(fig, full_html=False)

            # llm part
            prompt = SMA_interpretation_prompt.format(
                window=window, last_sma=last_sma, last_price=last_price,
                price_above_sma=price_above_sma, sma_trend_slope=slope,
                price_sma_perc_diff=perc_diff, recent_cross_ups=int(cross_ups),
                recent_cross_downs=int(cross_downs)
            )

            response = self.llm_gpt_o4.invoke([{"role": "system", "content": prompt}])
            interpretation = response.content

            self.crypto_chart_list.append({"chart_id": "indicator_SMA",
                                           "chart_name": "SMA analysis",
                                           "chart_html": chart_html,
                                           "plotly_fig": fig,
                                           "coin": crypto_name,
                                           "interpretation": interpretation})

            logging.info(f"Created an SMA chart for {crypto_name} between {min_date} and {max_date}")

            return {
                "indicator": "SMA",
                "params": {"window": window},
                "price_summary": price_data.get("price summary"),
                "sma_summary": summary
            }
        except Exception as e:
            error_message = f"Error creating chart: {e}"
            raise error_message

    @track_errors
    def get_ema_chart(self, crypto_name: str, min_date: str, max_date: str, span: int = 20) -> dict:
        """
        Generate a price chart with an Exponential Moving Average (EMA).
        The EMA gives more weight to recent prices compared to the SMA.
        Interpretation:
         - Useful for tracking short-term trend changes.
         - Price above EMA may indicate bullish bias; below EMA may indicate bearish bias.
         - EMA crossovers (short vs. long) often used as trading signals.

        Parameters
        ----------
        crypto_name : str
            Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".
        span : int, optional
            Span parameter for pandas `ewm`. Default 20.

        Returns
        -------
        dict
            - "chart_html": str
            - "indicator": "EMA"
            - "params": {"span": span}
            - "price_summary": dict

        Error handling
        --------------
        Returns {"error": "message"} on failure.
        """
        try:
            price_data = self.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)
            if "error" in price_data:
                return price_data

            df = self._make_price_df(price_data)
            df[f"EMA_{span}"] = df["price"].ewm(span=span, adjust=False).mean()

            dates = df.index.strftime("%Y-%m-%d").tolist()

            fig = Figure()
            fig.add_trace(Scatter(x=dates, y=df["price"], mode="lines", name="Price"))
            fig.add_trace(Scatter(x=dates, y=df[f"EMA_{span}"], mode="lines", name=f"EMA {span}"))
            fig.update_layout(title=f"{crypto_name.upper()} Price + EMA{span}", xaxis_title="Date",
                              yaxis_title="Price (USD)", template="plotly_white", height=420)

            ema_col = f"EMA_{span}"
            ema = df[ema_col]
            price = df["price"]

            last_ema = ema.iloc[-1]
            last_price = price.iloc[-1]

            # Position of price relative to EMA
            price_above_ema = last_price > last_ema

            # EMA trend slope over last span periods
            recent_ema = ema.iloc[-span:]
            slope, _, _, _, _ = linregress(np.arange(len(recent_ema)), recent_ema)

            # Percentage difference between price and EMA
            perc_diff = ((last_price - last_ema) / last_ema) * 100 if last_ema != 0 else 0

            summary = {
                "last_ema": last_ema,
                "last_price": last_price,
                "price_above_ema": price_above_ema,
                "ema_trend_slope": slope,
                "price_ema_perc_diff": perc_diff
            }

            chart_html = pio.to_html(fig, full_html=False)
            logging.info(f"Chart generated ({ema_col}): {truncate_string(chart_html)}")

            # llm part
            prompt = EMA_interpretation_prompt.format(
                span=span, last_ema=last_ema, last_price=last_price, price_above_ema=price_above_ema,
                ema_trend_slope=slope, price_ema_perc_diff=perc_diff,
            )

            response = self.llm_gpt_o4.invoke([{"role": "system", "content": prompt}])
            interpretation = response.content

            self.crypto_chart_list.append({"chart_id": "indicator_EMA",
                                           "chart_name": "EMA analysis",
                                           "chart_html": chart_html,
                                           "plotly_fig": fig,
                                           "coin": crypto_name,
                                           "interpretation": interpretation})

            logging.info(f"Created a EMA chart for {crypto_name} between {min_date} and {max_date}.")

            return {
                "indicator": "EMA",
                "params": {"span": span},
                "price_summary": price_data.get("price summary"),
                "ema_summary": summary
            }
        except Exception as e:
            error_message = f"Error creating chart: {e}"
            raise error_message

    def _rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Wilder smoothing
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        # For first window periods, fallback to simple mean
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.bfill().fillna(0)

    @track_errors
    def get_rsi_chart(self, crypto_name: str, min_date: str, max_date: str, window: int = 14) -> dict:
        """
        Generate a two-panel chart: price (top) and RSI (bottom).
        RSI measures momentum based on recent price gains vs. losses, values range 0–100.
        Interpretation:
         - RSI > 70 may suggest overbought conditions (potential reversal down).
         - RSI < 30 may suggest oversold conditions (potential reversal up).
         - Divergences between RSI and price can indicate weakening trends.

        Parameters
        ----------
        crypto_name : str
            Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".
        window : int, optional
            RSI window (default: 14).

        Returns
        -------
        dict
            - "chart_html": str (plotly)
            - "indicator": "RSI"
            - "params": {"window": window}
            - "price_summary": dict

        Notes
        -----
        - RSI values are scaled 0-100. Horizontal lines at 30 and 70 are plotted for context.
        - If volume/history is missing or too short, RSI uses available data and will still plot.
        """
        try:
            price_data = self.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)
            if "error" in price_data:
                return price_data

            df = self._make_price_df(price_data)
            df["RSI"] = self._rsi(df["price"], window=window)

            dates = df.index.strftime("%Y-%m-%d").tolist()

            # two-subplot figure (price + rsi)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(Scatter(x=dates, y=df["price"], mode="lines", name="Price"), row=1, col=1)
            fig.add_trace(Scatter(x=dates, y=df["RSI"], mode="lines", name=f"RSI {window}"), row=2, col=1)
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig.update_layout(title=f"{crypto_name.upper()} Price + RSI{window}", template="plotly_white", height=520)

            rsi = df["RSI"]

            last_rsi = rsi.iloc[-1]

            # Overbought/oversold flags
            overbought = last_rsi > 70
            oversold = last_rsi < 30

            # RSI trend (slope over last 'window' periods)
            recent_rsi = rsi.iloc[-window:]
            slope, _, _, _, _ = linregress(np.arange(len(recent_rsi)), recent_rsi)

            # Crossovers of 30 and 70 in last few periods
            def crossed(series, level):
                above = series > level
                crosses = above & ~above.shift(1, fill_value=False)
                return crosses.any()

            crossed_above_30 = crossed(rsi.iloc[-window:], 30)
            crossed_below_70 = crossed(rsi.iloc[-window:], 70)

            summary = {
                "last_rsi": last_rsi,
                "overbought": overbought,
                "oversold": oversold,
                "rsi_trend_slope": slope,
                "crossed_above_30_recently": crossed_above_30,
                "crossed_below_70_recently": crossed_below_70
            }

            chart_html = pio.to_html(fig, full_html=False)

            # llm part
            prompt = RSI_interpretation_prompt.format(
                last_rsi=last_rsi, overbought=overbought, oversold=oversold,
                rsi_trend_slope=slope, crossed_above_30_recently=crossed_above_30,
                crossed_below_70_recently=crossed_below_70
            )

            response = self.llm_gpt_o4.invoke([{"role": "system", "content": prompt}])
            interpretation = response.content

            self.crypto_chart_list.append({"chart_id": "indicator_RSI",
                                           "chart_name": "RSI analysis",
                                           "chart_html": chart_html,
                                           "plotly_fig": fig,
                                           "coin": crypto_name,
                                           "interpretation": interpretation})

            logging.info(f"Created a RSI chart for {crypto_name} between {min_date} and {max_date}.")

            return {
                "indicator": "RSI",
                "params": {"window": window},
                "price_summary": price_data.get("price summary"),
                "rsi_summary": summary
            }
        except Exception as e:
            error_message = f"Error creating chart: {e}"
            raise error_message

    @track_errors
    def get_macd_chart(self, crypto_name: str, min_date: str, max_date: str,
                       fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
        """
        Plot price together with the Moving Average Convergence Divergence (MACD) indicator.

        The MACD consists of two lines:
         - The MACD line (difference between short-term and long-term EMAs).
         - The signal line (EMA of the MACD line).

        Interpretation:
         - Crossovers between the MACD and the signal line can indicate possible trend shifts.
         - Values above 0 suggest bullish momentum; below 0 suggest bearish momentum.
         - This chart helps identify trend direction, strength, and potential reversals.

        Parameters
        ----------
        crypto_name : str
            Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".
        fast_period : int, optional
            Number of periods for the fast (short-term) EMA. Default: 12.
        slow_period : int, optional
            Number of periods for the slow (long-term) EMA. Default: 26.
        signal_period : int, optional
            Number of periods used to compute the signal line EMA from the MACD line. Default: 9.

        Returns
        -------
        dict
            - "chart_html": str
            - "indicator": "MACD"
            - "params": {"fast": 12, "slow": 26, "signal": 9}
            - "price_summary": dict

        Notes
        -----
        - MACD is computed as EMA(fast) - EMA(slow); signal is EMA(MACD).
        - A MACD histogram can be added if desired (bars).
        """

        try:
            price_data = self.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)
            if "error" in price_data:
                return price_data

            df = self._make_price_df(price_data)
            ema_fast = df["price"].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df["price"].ewm(span=slow_period, adjust=False).mean()
            df["MACD"] = ema_fast - ema_slow
            df["MACD_SIGNAL"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
            df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

            dates = df.index.strftime("%Y-%m-%d").tolist()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.7, 0.3], vertical_spacing=0.06)
            fig.add_trace(Scatter(x=dates, y=df["price"], mode="lines", name="Price"), row=1, col=1)
            fig.add_trace(Scatter(x=dates, y=df["MACD"], mode="lines", name="MACD"), row=2, col=1)
            fig.add_trace(Scatter(x=dates, y=df["MACD_SIGNAL"], mode="lines", name="Signal", line=dict(dash="dash")),
                          row=2, col=1)
            # optional histogram as bars: use bar traces if desired
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            fig.update_layout(title=f"{crypto_name.upper()} Price + MACD", template="plotly_white", height=560)

            macd = df["MACD"]
            signal = df["MACD_SIGNAL"]
            hist = df["MACD_HIST"]

            # last values
            last_macd = macd.iloc[-1]
            last_signal = signal.iloc[-1]
            last_hist = hist.iloc[-1]

            # zero-line position
            zero_pos = "above" if last_macd > 0 else "below" if last_macd < 0 else "on"

            # detect recent crossovers (in last 10 periods)
            crossovers = ((macd - signal).iloc[-10:] * (macd - signal).shift(1).iloc[-10:] < 0)
            bullish_cross = any(crossovers & ((macd - signal).iloc[-10:] > 0))
            bearish_cross = any(crossovers & ((macd - signal).iloc[-10:] < 0))

            # histogram peaks and valleys count
            peaks, _ = find_peaks(hist)
            valleys, _ = find_peaks(-hist)

            # slope of MACD trend
            slope, _, _, _, _ = linregress(np.arange(len(macd)), macd)

            summary = {
                "last_macd": last_macd,
                "last_signal": last_signal,
                "last_histogram": last_hist,
                "macd_zero_position": zero_pos,
                "bullish_crossover_recent": bullish_cross,
                "bearish_crossover_recent": bearish_cross,
                "histogram_peaks_count": len(peaks),
                "histogram_valleys_count": len(valleys),
                "macd_trend_slope": slope
            }

            chart_html = pio.to_html(fig, full_html=False)

            # llm part
            prompt = MACD_interpretation_prompt.format(
                last_macd=last_macd, last_signal=last_signal, last_histogram=last_hist,
                macd_zero_position=zero_pos, bullish_crossover_recent=bullish_cross,
                bearish_crossover_recent=bearish_cross, histogram_peaks_count=len(peaks),
                histogram_valleys_count=len(valleys), macd_trend_slope=slope
            )

            response = self.llm_gpt_o4.invoke([{"role": "system", "content": prompt}])
            interpretation = response.content

            self.crypto_chart_list.append({"chart_id": "indicator_MACD",
                                           "chart_name": "MACD analysis",
                                           "chart_html": chart_html,
                                           "plotly_fig": fig,
                                           "coin": crypto_name,
                                           "interpretation": interpretation})

            logging.info(f"Created a MACD chart for {crypto_name} between {min_date} and {max_date}.")

            return {
                "indicator": "MACD",
                "params": {"fast": fast_period, "slow": slow_period, "signal": signal_period},
                "price_summary": price_data.get("price summary"),
                "macd_summary": summary
            }
        except Exception as e:
            error_message = f"Error creating chart: {e}"
            raise error_message

    @track_errors
    def get_bollinger_bands_chart(self, crypto_name: str, min_date: str, max_date: str,
                                  window: int = 20, n_std: float = 2.0) -> dict:
        """
        Generate a price chart with Bollinger Bands (SMA ± n_std * standard deviation).
        Bollinger Bands show volatility around the moving average:
         - The middle band is a Simple Moving Average (SMA).
         - The upper/lower bands expand and contract with volatility.
        Interpretation:
         - Price touching or moving outside bands may signal overbought/oversold conditions.
         - Band squeezes (narrowing) often precede volatility breakouts.

        Parameters
        ----------
        crypto_name : str
            Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".
        window : int, optional
        n_std : float, optional

        Returns
        -------
        dict
            - "chart_html": str
            - "indicator": "BollingerBands"
            - "params": {"window": 20, "n_std": 2.0}
            - "price_summary": dict
        """
        try:
            price_data = self.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)
            if "error" in price_data:
                return price_data

            df = self._make_price_df(price_data)
            df[f"SMA_{window}"] = df["price"].rolling(window=window, min_periods=1).mean()
            df[f"STD_{window}"] = df["price"].rolling(window=window, min_periods=1).std()
            df[f"BB_UPPER"] = df[f"SMA_{window}"] + (n_std * df[f"STD_{window}"])
            df[f"BB_LOWER"] = df[f"SMA_{window}"] - (n_std * df[f"STD_{window}"])

            dates = df.index.strftime("%Y-%m-%d").tolist()

            fig = Figure()
            fig.add_trace(Scatter(x=dates, y=df["price"], mode="lines", name="Price"))
            fig.add_trace(Scatter(x=dates, y=df[f"SMA_{window}"], mode="lines", name=f"SMA {window}"))
            fig.add_trace(Scatter(x=dates, y=df["BB_UPPER"], mode="lines", name="BB Upper", line=dict(dash="dash")))
            fig.add_trace(Scatter(x=dates, y=df["BB_LOWER"], mode="lines", name="BB Lower", line=dict(dash="dash")))
            fig.update_layout(title=f"{crypto_name.upper()} Bollinger Bands", xaxis_title="Date",
                              yaxis_title="Price (USD)",
                              template="plotly_white", height=420)

            # Calculate bandwidth (volatility)
            df["BB_Width"] = df["BB_UPPER"] - df["BB_LOWER"]

            # Calculate %b to locate price within bands (0 = lower band, 1 = upper band)
            df["PercentB"] = (df["price"] - df["BB_LOWER"]) / df["BB_Width"]

            # Identify if price is above upper or below lower band recently
            df["Above_Upper"] = df["price"] > df["BB_UPPER"]
            df["Below_Lower"] = df["price"] < df["BB_LOWER"]

            # Add band squeeze detection: small width indicates squeeze
            squeeze_threshold = df["BB_Width"].quantile(0.25)  # adjust threshold as needed
            df["Squeeze"] = df["BB_Width"] <= squeeze_threshold

            # Summary (last few periods)
            summary = {
                "last_price": df["price"].iloc[-1],
                "last_sma": df[f"SMA_{window}"].iloc[-1],
                "last_upper_band": df["BB_UPPER"].iloc[-1],
                "last_lower_band": df["BB_LOWER"].iloc[-1],
                "last_percent_b": df["PercentB"].iloc[-1],
                "bands_width_mean": df["BB_Width"].mean(),
                "squeeze_recent": df["Squeeze"].iloc[-window:].any(),
                "price_outside_bands": {
                    "above_upper": df["Above_Upper"].sum(),
                    "below_lower": df["Below_Lower"].sum()
                }
            }

            chart_html = pio.to_html(fig, full_html=False)

            # llm part
            prompt = Bollinger_Bands_interpretation_prompt.format(
                last_price=summary["last_price"], last_sma=summary["last_sma"],
                last_upper_band=summary["last_upper_band"], last_lower_band=summary["last_lower_band"],
                last_percent_b=summary["last_percent_b"], bands_width_mean=summary["bands_width_mean"],
                squeeze_recent=summary["squeeze_recent"],
                price_outside_bands_above_upper=summary["price_outside_bands"]["above_upper"],
                price_outside_bands_below_lower=summary["price_outside_bands"]["below_lower"])

            response = self.llm_gpt_o4.invoke([{"role": "system", "content": prompt}])
            interpretation = response.content

            self.crypto_chart_list.append({"chart_id": "indicator_BollingerBands",
                                           "chart_name": "Bollinger Bands analysis",
                                           "chart_html": chart_html,
                                           "plotly_fig": fig,
                                           "coin": crypto_name,
                                           "interpretation": interpretation})

            logging.info(f"Created a bollinger bands chart for {crypto_name} between {min_date} and {max_date}.")

            return {
                "indicator": "BollingerBands",
                "params": {"window": window, "n_std": n_std},
                "price_summary": price_data.get("price summary"),
                "bollinger_summary": summary,
            }
        except Exception as e:
            error_message = f"Error creating chart: {e}"
            raise error_message

    @track_errors
    def get_obv_chart(self, crypto_name: str, min_date: str, max_date: str) -> dict:
        """
        Generate a price chart with On-Balance Volume (OBV).
        OBV combines price and volume to measure buying/selling pressure.
        Interpretation:
         - Rising OBV confirms upward price momentum (buyers dominating).
         - Falling OBV confirms downward price momentum (sellers dominating).
         - Divergences (price rising but OBV flat/falling) may warn of trend weakness.

        Parameters
        ----------
        crypto_name : str
            Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".

        Returns
        -------
        dict
            - "chart_html": str
            - "indicator": "OBV"
            - "params": {"crypto_name": bitcoin, "min_date": YYYY-MM-DD}
            - "price_summary": dict

        Notes
        -----
        - OBV is computed as cumulative signed volume: add volume when price rises, subtract when price falls.
        - Volume must be present in the fetched data for meaningful OBV.
        """
        try:
            price_data = self.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)
            if "error" in price_data:
                return price_data

            df = self._make_price_df(price_data)

            # OBV: +volume when price up, -volume when price down
            df["price_diff"] = df["price"].diff().fillna(0)
            df["volume_signed"] = np.where(df["price_diff"] > 0, df["volume"], -df["volume"])
            df["OBV"] = df["volume_signed"].fillna(0).cumsum()

            dates = df.index.strftime("%Y-%m-%d").tolist()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
            fig.add_trace(Scatter(x=dates, y=df["price"], mode="lines", name="Price"), row=1, col=1)
            fig.add_trace(Scatter(x=dates, y=df["OBV"], mode="lines", name="OBV"), row=2, col=1)
            fig.update_layout(title=f"{crypto_name.upper()} Price + OBV", template="plotly_white", height=520)

            obv = df["OBV"]
            price = df["price"]

            obv_trend = obv.iloc[-1] - obv.iloc[0]
            price_trend = price.iloc[-1] - price.iloc[0]

            # Detect divergence
            divergence = (obv_trend * price_trend) < 0

            # Calculate OBV slope via linear regression
            slope, _, _, _, _ = linregress(np.arange(len(obv)), obv)

            # OBV position relative to history
            obv_level = (obv.iloc[-1] - obv.min()) / (obv.max() - obv.min()) if obv.max() != obv.min() else 0.5

            # Significant changes (e.g., volume spikes)
            obv_diff = obv.diff().abs()
            spikes = (obv_diff > obv_diff.quantile(0.9)).sum()

            summary = {
                "obv_trend": obv_trend,
                "price_trend": price_trend,
                "divergence": divergence,
                "obv_slope": slope,
                "obv_level": obv_level,
                "spikes": spikes
            }

            chart_html = pio.to_html(fig, full_html=False)

            # llm part
            prompt = OBV_interpretation_prompt.format(obv_trend=obv_trend, price_trend=price_trend,
                                                      divergence=divergence, obv_slope=slope, obv_level=obv_level,
                                                      spikes=spikes)
            response = self.llm_gpt_o4.invoke([{"role": "system", "content": prompt}])
            interpretation = response.content

            self.crypto_chart_list.append({"chart_id": "indicator_OBV",
                                           "chart_name": "OBV analysis",
                                           "chart_html": chart_html,
                                           "plotly_fig": fig,
                                           "coin": crypto_name,
                                           "interpretation": interpretation})

            logging.info(f"Created a OBV chart for {crypto_name} between {min_date} and {max_date}")

            return {
                "indicator": "OBV",
                "params": {},
                "price_summary": price_data.get("price summary"),
                "obv_summary": summary,
            }
        except Exception as e:
            error_message = f"get_obv_chart error: {e}"
            raise error_message

    def compare_cryptos(
            self,
            crypto_coins_to_compare: List[str],
            min_date: str,
            max_date: str,
            chart: bool = True,
            max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Compare up to 4 cryptocurrencies over a date range, aligning y-axes visually
        so both price scales have consistent grid spacing even when values differ.

        Returns
        -------
        dict
            {
              "coins_compared": [...],
              "summaries": [ {coin summary dict}, ... ],
              "errors": { coin_name: "error msg", ... }
            }
        """
        # Basic validation
        if not crypto_coins_to_compare:
            return {"error": "No coins provided to compare."}
        if len(crypto_coins_to_compare) > 4:
            return {"error": "You can only compare up to 4 coins."}

        coins = [c.strip().lower() for c in crypto_coins_to_compare]

        # Parallel fetch
        futures, results, errors = {}, {}, {}

        def _fetch(crypto_coin):
            try:
                raw_price_data = self.wrapped_get_crypto_price_range(crypto_coin, min_date, max_date)
                return crypto_coin, raw_price_data
            except Exception as exc:
                return crypto_coin, {"error": f"fetch exception: {exc}"}

        with ThreadPoolExecutor(max_workers=min(max_workers, len(coins))) as ex:
            for coin in coins:
                futures[ex.submit(_fetch, coin)] = coin

            for fut in as_completed(futures):
                coin = futures[fut]
                try:
                    _, price_data = fut.result()
                except Exception as e:
                    errors[coin] = f"Exception fetching: {e}"
                    continue

                if "error" in price_data:
                    errors[coin] = price_data["error"]
                    continue

                try:
                    df = self._make_price_df(price_data)[["price"]].rename(columns={"price": coin})
                    results[coin] = {"df": df, "summary": price_data.get("price summary", {})}
                except Exception as e:
                    errors[coin] = f"Failed to build df: {e}"
                    continue

        if not results:
            return {"error": "No valid price data fetched for any requested coin.", "errors": errors}

        # Align all DataFrames on date index
        all_dfs = [v["df"] for v in results.values()]
        combined = pd.concat(all_dfs, axis=1, join="outer").sort_index()
        combined = combined.ffill().bfill()

        # Compute summaries
        summaries = []
        for coin, meta in results.items():
            series = combined[coin].dropna()
            if series.empty:
                summaries.append({"coin": coin, "error": "no data after alignment"})
                continue
            start_price = float(series.iloc[0])
            end_price = float(series.iloc[-1])
            price_change = round(end_price - start_price, 8)
            pct_change = round(((end_price - start_price) / start_price) * 100, 4) if start_price != 0 else None
            summaries.append({
                "coin": coin,
                "start_date": series.index[0].strftime("%Y-%m-%d"),
                "end_date": series.index[-1].strftime("%Y-%m-%d"),
                "start_price": start_price,
                "end_price": end_price,
                "min_price": float(series.min()),
                "max_price": float(series.max()),
                "average_price": float(series.mean()),
                "price_change": price_change,
                "percentage_change": pct_change,
                "price_summary_raw": results[coin]["summary"]
            })

        charts_html = []

        if chart:
            for coin_a, coin_b in combinations(combined.columns, 2):
                fig = Figure()
                dates = combined.index.strftime("%Y-%m-%d").tolist()

                # Add first trace (left y-axis)
                fig.add_trace(Scatter(
                    x=dates,
                    y=combined[coin_a],
                    mode="lines",
                    name=coin_a.upper(),
                    yaxis="y1"
                ))

                # Add second trace (right y-axis)
                fig.add_trace(Scatter(
                    x=dates,
                    y=combined[coin_b],
                    mode="lines",
                    name=coin_b.upper(),
                    yaxis="y2"
                ))

                # --- Dynamic Tick Alignment (Solution 3) ---
                y1_min, y1_max = combined[coin_a].min(), combined[coin_a].max()
                y2_min, y2_max = combined[coin_b].min(), combined[coin_b].max()

                num_ticks = 8  # number of major gridlines
                y1_ticks = np.linspace(y1_min, y1_max, num_ticks)
                y2_ticks = np.linspace(y2_min, y2_max, num_ticks)

                fig.update_layout(
                    title=f"{coin_a.upper()} vs {coin_b.upper()}",
                    xaxis=dict(title="Date"),
                    yaxis=dict(
                        title=f"{coin_a.upper()} Price (USD)",
                        side="left",
                        tickvals=y1_ticks,
                        tickmode="array"
                    ),
                    yaxis2=dict(
                        title=f"{coin_b.upper()} Price (USD)",
                        overlaying="y",
                        side="right",
                        tickvals=y2_ticks,
                        tickmode="array"
                    ),
                    template="plotly_white",
                    height=480,
                    legend=dict(x=0.02, y=1.10),
                )
                # --------------------------------------------

                chart_html = pio.to_html(fig, full_html=False)
                charts_html.append({"pair": (coin_a, coin_b), "html": chart_html})

                self.crypto_chart_list.append({
                    "chart_id": f"comparison_{coin_a}_{coin_b}",
                    "chart_name": f"{coin_a.upper()} vs {coin_b.upper()}",
                    "chart_html": chart_html,
                    "plotly_fig": fig,
                    "coins_compared": [coin_a, coin_b]
                })

            logging.info(f"Created pairwise comparison charts for {crypto_coins_to_compare} "
                         f"between {min_date} and {max_date}.")

        return {
            "coins_compared": list(combined.columns),
            "summaries": summaries,
            "errors": errors or None
        }

    def wrapped_get_crypto_price_range(self, crypto_name: str, min_date: str, max_date: str) -> dict:
        """
        Wrapper function for fetching price data of a cryptocurrency in USD over a date range. Also used for general
        price analysis or trends for a supported coin over a period. If the range period is older than permitted,
        it gets adjusted. If it can't be adjusted, or it is too old, returns an error in a dict.

        Parameters
        ----------
        crypto_name : str
            Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD".

        Returns
        -------
        dict
        """
        # Demo can only fetch price date from the past year, this is a limit/ graceful error handling
        date_today = dt.datetime.now()
        oldest_crypto_price_date_possible = date_today - dt.timedelta(days=CryptoValues.HISTORY_RANGE.value)
        min_date_extracted_str = min_date
        max_date_extracted_str = max_date

        try:
            min_date_extracted = dt.datetime.strptime(min_date_extracted_str, "%Y-%m-%d")
            max_date_extracted = dt.datetime.strptime(max_date_extracted_str, "%Y-%m-%d")

        except ValueError:
            error_message = (f"The provided dates for price range fetching are not in the correct format "
                             f"(YYYY-MM-DD). Min date: {min_date_extracted_str}, "
                             f"max date: {max_date_extracted_str}")
            raise error_message

        if min_date_extracted >= oldest_crypto_price_date_possible:
            return self.get_crypto_price_range(crypto_name, min_date, max_date)

        elif max_date_extracted > oldest_crypto_price_date_possible:
            min_date = oldest_crypto_price_date_possible.strftime("%Y-%m-%d")
            return self.get_crypto_price_range(crypto_name, min_date, max_date)

        else:
            return {
                "error": f'Can only fetch price data from the last {CryptoValues.HISTORY_RANGE.value} days. '
                         f'The date range from {min_date_extracted_str} to {max_date_extracted_str} is too old.'
            }

    def wrapped_get_crypto_news(self, crypto_name: str, min_date: str, max_date: str = None) -> dict:
        """
        Wrapper function for fetching news articles about a given cryptocurrency from a given date range. If the range
        period is older than permitted, it gets adjusted. If it can't be adjusted, or it is too old, returns an error in
        a dict.

        Parameters
        ----------
        crypto_name : str
            Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
        min_date : str
            Start date in "YYYY-MM-DD".
        max_date : str
            End date in "YYYY-MM-DD". (optional)

        Returns
        -------
        dict
        """
        # Demo can only fetch news from the past thirty days, this is a limit/ graceful error handling
        date_today = dt.datetime.now()
        oldest_news_date_possible = date_today - dt.timedelta(days=CryptoValues.NEWS_RANGE.value)
        min_date_extracted_str = min_date
        max_date_extracted_str = max_date

        min_date_extracted = dt.datetime.strptime(min_date_extracted_str, "%Y-%m-%d")
        max_date_extracted = dt.datetime.strptime(max_date_extracted_str, "%Y-%m-%d")
        if min_date_extracted >= oldest_news_date_possible:
            return self.get_crypto_news(crypto_name, min_date, max_date)

        elif max_date_extracted and max_date_extracted > oldest_news_date_possible:
            min_date = oldest_news_date_possible.strftime("%Y-%m-%d")
            return self.get_crypto_news(crypto_name, min_date, max_date)

        else:
            return {"error": f'Can only fetch news from the last {CryptoValues.NEWS_RANGE.value}  days. '
                             f'The date range from {min_date_extracted_str} to {max_date_extracted_str} '
                             f'is too old.'}

    @track_errors
    def crypto_faq(self, crypto_name: str, user_query: str) -> dict:
        """
        A functions that uses AI to answer general and trivia questions about supported cryptocurrencies.

        Parameters
        ----------
        crypto_name : str
            Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
        user_query : str
            The user's query

        Returns
        -------
        dict
        """

        tools = {_tool.name: (_tool.__doc__ or "No description provided.") for _tool in self.tools}
        faq_prompt = langgraph_faq_prompt.format(query=user_query, crypto_name=crypto_name, tools=tools)
        try:
            response = self.llm_gpt_o4.invoke([
                {"role": "system", "content": faq_prompt},
            ])
            faq_response = response.content
            return {"user_query": self.user_query, "faq_answer": faq_response}
        except Exception as e:
            error_message = f"Failed to get an faq answer for {self.user_query}, error: {e}"
            raise error_message

    def clear_previous_charts(self):
        """
        Clear previously used charts from a list to add new ones
        """
        self.crypto_chart_list = []

    def tool_selection_llm(self, state: AgentState):
        self.clear_previous_charts()

        try:
            tool_prompt = tool_selection_prompt.format(date_today=self.date_today,
                                                       user_query=self.user_query,
                                                       supported_coin_list=self.supported_coin_list)

            llm_instance = self.llm_gpt_o4.bind_tools(tools=self.tools)
            success_message = f"[tool_selection_llm] Successfully bound tools: {[t.name for t in self.tools]}"
            logging.info(success_message)
            self.llm_with_tools = llm_instance
            result = llm_instance.invoke(tool_prompt)

            return {"messages": [result], "tool_prompt": tool_prompt}

        except Exception as e:
            logging.error(f"[tool_selection] Unexpected error: {e}")
            logging.debug(traceback.format_exc())
            return {"error": str(e)}

    def tool_metadata_collector(self, state: AgentState) -> dict:
        """
        Collect tool usage metadata right after the LLM decides which tools to call.
        """
        tools_used = []
        for msg in state.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    tools_used.append({
                        "name": call.get("name", "no tool name found"),
                        "args": call.get("args", {})
                    })

        return {"used_tools": tools_used}

    def get_ai_answer(self, state: AgentState):
        """
        Processes a prompt to send it to the llm used for the langgraph agent

        Parameters
        ----------
        state: AgentState
            Langgraph state class

        Returns
        -------
        dict
        """
        state_messages = state.get("messages", "there are no tool outputs")
        used_tools = state.get("used_tools", "no tools were used")
        used_tools_name_only = [_tool.get("name", "") for _tool in used_tools]
        # remove duplicates
        used_tools_name_only = list(dict.fromkeys(used_tools_name_only))

        logging.info(f"Tools used: {used_tools_name_only}")

        for chart in self.crypto_chart_list:
            if chart.get("chart_html", None):
                logging.info(f"This chart has an html key: {chart["chart_id"]}")
                continue
            else:
                logging.warning(f"This chart has no html key: {chart}")

        final_prompt = result_synthesis_prompt.format(
            tool_outputs=state_messages,
            user_query=self.user_query,
            supported_coin_list=self.supported_coin_list)

        try:
            response = self.llm_with_tools.invoke([{"role": "system", "content": final_prompt}])
        except Exception as e:
            return {"error": f"There was an error generating AI response: {e}]"}

        logging.info(f"Final answer ready.")
        return {"messages": [response], "final_prompt": final_prompt}

    def graph_debugger(self, state: AgentState):
        if self.debugging:
            ai_replies = [f"Reply {i}: {reply}" for i, reply in enumerate(state.get('messages'), start=1)]
            tool_prompt = state.get("tool_prompt")
            final_prompt = state.get("final_prompt")
            used_tools = state.get("used_tools")
            logging.info(f"Tool_prompt used: {tool_prompt}\n")
            logging.info(f"AI responses: {'\n'.join(ai_replies)}\n")
            logging.info(f"Final_prompt used: {final_prompt}\n")
            logging.info(f"Tools used: {used_tools}\n")
        return state

    # Function to run the agent
    def stream_graph_updates(self, user_query: str):
        self.user_query = user_query
        last_message = ""
        for event in self.graph.stream({"messages": [{"role": "user", "content": user_query}]}):
            for value in event.values():
                if isinstance(value, dict) and "messages" in value and value["messages"]:
                    try:
                        last_msg = value["messages"][-1]

                        # Handle dict or object
                        if isinstance(last_msg, dict):
                            content = last_msg.get("content", "")
                        else:
                            # It's likely a LangChain/Graph message object
                            content = getattr(last_msg, "content", "")

                        logging.info(f"Assistant: {content}")
                        last_message = content
                    except Exception as e:
                        logging.error(f"Failed to read message content: {e}")
                else:
                    logging.debug(f"Skipping event without messages: {value}")
        return last_message

    def get_graph_mermaid_syntax(self):
        try:
            print(self.graph.get_graph().draw_mermaid())
            display(Image(self.graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)))
        except Exception as e:
            logging.error(f"Error occurred, couldn't display the graph: {e}")

    def _build_graph(self):
        tool_node = ToolNode(tools=self.tools)

        graph_builder = StateGraph(AgentState)

        # Nodes
        graph_builder.add_node("tool_selection_llm", self.tool_selection_llm)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_node("metadata_collection", self.tool_metadata_collector)
        graph_builder.add_node("debugger", self.graph_debugger)
        graph_builder.add_node("ai_crypto_analyst", self.get_ai_answer)

        # Flow
        graph_builder.set_entry_point("tool_selection_llm")
        graph_builder.add_edge("tool_selection_llm", "tools")
        graph_builder.add_edge("tools", "metadata_collection")
        graph_builder.add_edge("metadata_collection", "debugger")
        graph_builder.add_edge("debugger", "ai_crypto_analyst")
        graph_builder.add_edge("ai_crypto_analyst", END)

        graph = graph_builder.compile()
        return graph


@tool
def _wrapped_get_crypto_price_range(crypto_name: str, min_date: str, max_date: str) -> dict:
    """
    Wrapper function for fetching price data of a cryptocurrency in USD over a date range. Also used for general
    price analysis or trends for a supported coin over a period. If the range period is older than permitted,
    it gets adjusted. If it can't be adjusted, or it is too old, returns an error in a dict.

    Parameters
    ----------
    crypto_name : str
        Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".

    Returns
    -------
    dict
    """
    return crypto_ai_agent.wrapped_get_crypto_price_range(crypto_name, min_date, max_date)


@tool
def _wrapped_get_crypto_news(crypto_name: str, min_date: str, max_date: str = None) -> dict:
    """
    Wrapper function for fetching news articles about a given cryptocurrency from a given date range. If the range
    period is older than permitted, it gets adjusted. If it can't be adjusted, or it is too old, returns an error in
    a dict.

    Parameters
    ----------
    crypto_name : str
        Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD". (optional)

    Returns
    -------
    dict
    """
    return crypto_ai_agent.wrapped_get_crypto_news(crypto_name, min_date, max_date)


@tool
def _get_basic_crypto_price_chart(crypto_name: str, min_date: str, max_date: str) -> dict:
    """
    Creates a basic crypto price chart from the min_date to the max_date for the given crypto_name

    Parameters
    ----------
    crypto_name : str
        crypto_name: Name of the cryptocurrency, (e.g., 'bitcoin', 'ethereum').
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".

    Returns
    -------
    dict
    """
    return crypto_ai_agent.get_basic_crypto_price_chart(crypto_name, min_date, max_date)


@tool
def _get_capitalization_chart(crypto_name: str, min_date: str, max_date: str) -> dict:
    """
    Creates a crypto market capitalization chart from the min_date to the max_date for the given crypto_name

    Parameters
    ----------
    crypto_name : str
        crypto_name: Name of the cryptocurrency, (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".

    Returns
    -------
    dict
    """
    return crypto_ai_agent.get_capitalization_chart(crypto_name, min_date, max_date)


@tool
def _get_volume_chart(crypto_name: str, min_date: str, max_date: str) -> dict:
    """
    Creates a total volume chart from the min_date to the max_date for the given crypto_name

    Parameters
    ----------
    crypto_name : str
        crypto_name: Name of the cryptocurrency, (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', 'Tether').
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".

    Returns
    -------
    dict
    """
    return crypto_ai_agent.get_volume_chart(crypto_name, min_date, max_date)


@tool
def _get_sma_chart(crypto_name: str, min_date: str, max_date: str, window: int = 20) -> dict:
    """
    Generate a price chart with a Simple Moving Average (SMA).
    The SMA smooths out price data over a chosen period.
    Interpretation:
     - Helps identify overall trend direction.
     - Price above SMA may indicate uptrend; below SMA may indicate downtrend.
     - Longer SMAs (e.g., 200-day) act as support/resistance in many trading strategies.

    Parameters
    ----------
    crypto_name : str
        Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".
    window : int, optional
        Rolling window length for SMA. Default is 20.

    Returns
    -------
    dict
        JSON-serializable dict containing:
        - "chart_html": str (Plotly HTML fragment, safe to render via Jinja)
        - "indicator": "SMA"
        - "params": dict (e.g., {"window": 20})
        - "price_summary": dict (from the price fetcher) or None

    Error handling
    --------------
    Returns {"error": "message"} on failure.
    """
    return crypto_ai_agent.get_sma_chart(crypto_name, min_date, max_date, window)


@tool
def _get_ema_chart(crypto_name: str, min_date: str, max_date: str, span: int = 20) -> dict:
    """
    Generate a price chart with an Exponential Moving Average (EMA).
    The EMA gives more weight to recent prices compared to the SMA.
    Interpretation:
     - Useful for tracking short-term trend changes.
     - Price above EMA may indicate bullish bias; below EMA may indicate bearish bias.
     - EMA crossovers (short vs. long) often used as trading signals.

    Parameters
    ----------
    crypto_name : str
        Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".
    span : int, optional
        Span parameter for pandas `ewm`. Default 20.

    Returns
    -------
    dict
        - "chart_html": str
        - "indicator": "EMA"
        - "params": {"span": span}
        - "price_summary": dict

    Error handling
    --------------
    Returns {"error": "message"} on failure.
    """
    return crypto_ai_agent.get_ema_chart(crypto_name, min_date, max_date, span)


@tool
def _get_rsi_chart(crypto_name: str, min_date: str, max_date: str, window: int = 14) -> dict:
    """
    Generate a two-panel chart: price (top) and RSI (bottom).
    RSI measures momentum based on recent price gains vs. losses, values range 0–100.
    Interpretation:
     - RSI > 70 may suggest overbought conditions (potential reversal down).
     - RSI < 30 may suggest oversold conditions (potential reversal up).
     - Divergences between RSI and price can indicate weakening trends.

    Parameters
    ----------
    crypto_name : str
        Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".
    window : int, optional
        RSI window (default: 14).

    Returns
    -------
    dict
        - "chart_html": str (plotly)
        - "indicator": "RSI"
        - "params": {"window": window}
        - "price_summary": dict

    Notes
    -----
    - RSI values are scaled 0-100. Horizontal lines at 30 and 70 are plotted for context.
    - If volume/history is missing or too short, RSI uses available data and will still plot.
    """
    return crypto_ai_agent.get_rsi_chart(crypto_name, min_date, max_date, window)


@tool
def _get_macd_chart(crypto_name: str, min_date: str, max_date: str,
                    fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
    """
    Plot price together with the Moving Average Convergence Divergence (MACD) indicator.
    The MACD consists of two lines:
     - The MACD line (difference between short-term and long-term EMAs).
     - The signal line (EMA of the MACD line).
    Interpretation:
     - Crossovers between the MACD and the signal line can indicate possible trend shifts.
     - Values above 0 suggest bullish momentum; below 0 suggest bearish momentum.
    This chart is useful for spotting trend direction, strength, and potential reversals.

    Parameters
    ----------
    crypto_name : str
        Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".
    fast_period : int, optional
    slow_period : int, optional
    signal_period : int, optional

    Returns
    -------
    dict
        - "chart_html": str
        - "indicator": "MACD"
        - "params": {"fast": 12, "slow": 26, "signal": 9}
        - "price_summary": dict

    Notes
    -----
    - MACD is computed as EMA(fast) - EMA(slow); signal is EMA(MACD).
    - A MACD histogram can be added if desired (bars).
    """
    return crypto_ai_agent.get_macd_chart(crypto_name, min_date, max_date, fast_period, slow_period, signal_period)


@tool
def _get_bollinger_bands_chart(crypto_name: str, min_date: str, max_date: str,
                               window: int = 20, n_std: float = 2.0) -> dict:
    """
    Generate a price chart with Bollinger Bands (SMA ± n_std * standard deviation).
    Bollinger Bands show volatility around the moving average:
     - The middle band is a Simple Moving Average (SMA).
     - The upper/lower bands expand and contract with volatility.
    Interpretation:
     - Price touching or moving outside bands may signal overbought/oversold conditions.
     - Band squeezes (narrowing) often precede volatility breakouts.

    Parameters
    ----------
    crypto_name : str
        Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".
    window : int, optional
    n_std : float, optional

    Returns
    -------
    dict
        - "chart_html": str
        - "indicator": "BollingerBands"
        - "params": {"window": 20, "n_std": 2.0}
        - "price_summary": dict
    """
    return crypto_ai_agent.get_bollinger_bands_chart(crypto_name, min_date, max_date, window, n_std)


@tool
def _get_obv_chart(crypto_name: str, min_date: str, max_date: str) -> dict:
    """
    Generate a price chart with On-Balance Volume (OBV).
    OBV combines price and volume to measure buying/selling pressure.
    Interpretation:
     - Rising OBV confirms upward price momentum (buyers dominating).
     - Falling OBV confirms downward price momentum (sellers dominating).
     - Divergences (price rising but OBV flat/falling) may warn of trend weakness.

    Parameters
    ----------
    crypto_name : str
        Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Avalanche', Tether).
    min_date : str
        Start date in "YYYY-MM-DD".
    max_date : str
        End date in "YYYY-MM-DD".

    Returns
    -------
    dict
        - "chart_html": str
        - "indicator": "OBV"
        - "params": {"crypto_name": bitcoin, "min_date": YYYY-MM-DD}
        - "price_summary": dict

    Notes
    -----
    - OBV is computed as cumulative signed volume: add volume when price rises, subtract when price falls.
    - Volume must be present in the fetched data for meaningful OBV.
    """
    return crypto_ai_agent.get_obv_chart(crypto_name, min_date, max_date)


@tool
def _compare_cryptos(
        crypto_coins_to_compare: List[str],
        min_date: str,
        max_date: str,
        chart: bool = True,
        max_workers: int = 4
) -> Dict[str, Any]:
    """
    Compare up to 4 cryptocurrencies over a date range.

    Parameters
    ----------
    crypto_coins_to_compare : List[str]
        Names of coins to compare (e.g., ['bitcoin', 'ethereum']).
    min_date : str
        Start date "YYYY-MM-DD".
    max_date : str
        End date "YYYY-MM-DD".
    chart : bool, optional
        If True, produce Plotly HTML charts comparing the series pairwise.
    max_workers : int, optional
        Number of parallel fetch workers.

    Returns
    -------
    dict
        {
          "coins_compared": [...],
          "summaries": [ {coin summary dict}, ... ],
          "charts_html": [ {"pair": (coinA, coinB), "html": "<div>...</div>"}],
          "errors": { coin_name: "error msg", ... }
        }
    """
    return crypto_ai_agent.compare_cryptos(crypto_coins_to_compare, min_date, max_date, chart, max_workers)


@tool
def _crypto_faq(crypto_name: str, user_query: str):
    """
    A functions that uses AI to answer general and trivia questions about supported cryptocurrencies.

    Parameters
    ----------
    user_query : str
        The user's query

    Returns
    -------
    dict
    """
    crypto_ai_agent.crypto_faq(crypto_name, user_query)


crypto_ai_agent = CryptoAgent()


def truncate_string(string, max_length=100):
    if len(string) > max_length:
        return string[:max_length].replace(" ", "") + "' ... '" + string[-max_length:].replace(" ", "")


def console_test():
    agent = CryptoAgent()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if user_input.lower() == "show mermaid":
            agent.get_graph_mermaid_syntax()
            continue

        if user_input.lower() == "/debug-on":
            if not agent.debugging:
                agent.debugging = True
                logging.info("Langgraph agent debugging enabled.")

            else:
                logging.info("Langgraph agent debugging is already enabled.")

            continue

        elif user_input.lower() == "/debug-off":
            if agent.debugging:
                agent.debugging = False
                logging.info("Langgraph agent debugging disabled.")

            else:
                logging.info("Langgraph agent debugging is already disabled.")

            continue

        agent.stream_graph_updates(user_input)
        charts = agent.get_chart_list()
        charts_name_only = [chart["chart_id"] for chart in charts]
        charts_html_only = [chart["chart_html"] for chart in charts]
        logging.info(f"The number of charts made is {len(charts)}: {charts_name_only}\n"
                     f"The number of available chart htmls {len(charts_html_only)}.")


if __name__ == "__main__":
    console_test()
