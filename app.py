# 1️⃣ Standard Library Imports
import datetime as dt
import json
import os
import time
import traceback
import uuid
import numpy as np
import pandas as pd
import plotly.io as pio
import requests

# 2️⃣ Third-Party Library Imports
from flask import (
    render_template, request, redirect, url_for, flash,
    session
)
from flask_bootstrap import Bootstrap5
from flask_wtf.csrf import CSRFProtect
from plotly.graph_objs import Figure, Scatter
from sqlalchemy import delete
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# 3️⃣ Local Application Imports
from app_factory import create_app, logging
from database_models import db, CryptoMarketData
from enum_classes import CryptoCoins, CryptoValues
from forms import CryptoForm, CryptoFormAI
from langgraph_crypto_agent import crypto_ai_agent

# Initialize Flask
app = create_app()

# --- Configuration ---
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///crypto_data.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["HISTORY_RANGE"] = CryptoValues.HISTORY_RANGE.value

# Paths
SYS_TEMP_FOLDER = os.path.join(os.getcwd(), "temp_charts")
os.makedirs(SYS_TEMP_FOLDER, exist_ok=True)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")

# --- Extensions ---
csrf = CSRFProtect(app)
bootstrap = Bootstrap5(app)

# Scheduler setup
scheduler = BackgroundScheduler()


# Helper function: Get crypto data if there's none
def get_crypto_data(coin_name, coin_id):
    """Function that clears all existing crypto for a given coin and replaces it with price data from
    the last 30 days."""
    api_key = os.getenv("COIN_GECKO_API")
    endpoint = f"coins/{coin_id}/market_chart"
    url = f"https://api.coingecko.com/api/v3/{endpoint}"
    history_range = app.config.get("HISTORY_RANGE", 30)
    logging.info(f"Coin ID chosen after submission: {coin_id}")

    headers = {
        "accept": "application/json",
        "x-cg-api-key": api_key,
    }

    coin_in_db = db.session.query(CryptoMarketData).filter(CryptoMarketData.name == coin_name)

    # Clears previous data before adding new one
    if coin_in_db.all():
        coin_in_db.delete()
        db.session.commit()

    params = {"vs_currency": "usd", "days": history_range, "interval": "daily"}

    response = requests.get(url, headers=headers, params=params)

    max_retires = 5
    retry_delay = 15  # Start with a 15-second delay

    for attempt in range(max_retires):
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 429:
            logging.warning(f"Rate limit hit. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff (15s → 30s → 60s)
        else:
            break

    if "application/json" in response.headers["Content-Type"]:
        data = response.json()

        dates = [dt.datetime.fromtimestamp(ts // 1000, dt.timezone.utc) for ts, price in data['prices'][:-1]]
        price_data = [price for ts, price in data['prices'][:-1]]
        market_cap_data = [cap for ts, cap in data['market_caps'][:-1]]
        volume_data = [vol for ts, vol in data['total_volumes'][:-1]]

        new_entries = []

        coin_symbol = None

        list_of_coin_dicts = [coin.value for coin in CryptoCoins]
        for coin in list_of_coin_dicts:
            if coin['name'] == coin_name:
                coin_symbol = coin['symbol']
                break

        for date_, price, market_cap, volume in zip(dates, price_data, market_cap_data, volume_data):
            # logging.info(f"Date: {date_}, Price: {price:,.4f}, Market Cap: {market_cap:,.2f}, Volume: {volume:,.2f}")
            new_entries.append(CryptoMarketData(
                name=coin_name,
                symbol=coin_symbol,
                current_price=price,
                market_cap=market_cap,
                total_volume=volume,
                date=date_,
                image=f"assets/crypto/{coin_name}-icon.png",
            ))

        if new_entries:
            db.session.bulk_save_objects(new_entries)
            db.session.commit()
            logging.info(f"Added {len(new_entries)} new entries for {coin_name}.")
    else:
        logging.error("Received response is not JSON, cannot proceed.")


# Helper function: checks if timestamp is in a date list
def ts_in_date_list(ts, dates_: list):
    """Returns True if the timestamp is in the list of dates provided and False otherwise."""
    ts_date = dt.datetime.fromtimestamp(ts // 1000, dt.timezone.utc).date()
    return ts_date in dates_


# Helper function: Update crypto data rows
def update_crypto_data() -> None:
    """Updates price data to ensure price data from the determined history range are stored."""
    with ((app.app_context())):
        list_of_coin_dicts = [coin.value for coin in CryptoCoins]

        for coin in list_of_coin_dicts:
            coin_in_db = db.session.query(CryptoMarketData).filter(CryptoMarketData.name == coin['name']).all()

            if not coin_in_db:
                logging.info(f"No data for {coin['name']}. Fetching from scratch...")
                get_crypto_data(coin_name=coin["name"], coin_id=coin["id"])
                continue

            api_key = os.getenv("COIN_GECKO_API")
            url = f"https://api.coingecko.com/api/v3/coins/{coin['id']}/market_chart"
            history_range = app.config.get("HISTORY_RANGE", 30)

            headers = {"accept": "application/json", "x-cg-api-key": api_key}

            date_range_in_db = [row.date.date() for row in coin_in_db]
            oldest_history_range_date = dt.datetime.now() - dt.timedelta(days=history_range - 1)
            date_range_required = [(oldest_history_range_date + dt.timedelta(days=day)).date()
                                   for day in range(history_range)]
            missing_dates = [date_ for date_ in date_range_required if date_ not in date_range_in_db]

            if missing_dates:

                # Get historical data in one API call instead of multiple
                params = {"vs_currency": "usd", "days": history_range, "interval": "daily"}

                max_retries = 5
                retry_delay = 15  # Start with a 15-second delay

                for attempt in range(max_retries):
                    response = requests.get(url, headers=headers, params=params)

                    if response.status_code == 429:
                        logging.warning(f"Rate limit hit. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff (15s → 30s → 60s)
                    else:
                        break

                if "application/json" in response.headers["Content-Type"]:
                    data = response.json()

                    logging.info(f"Missing dates for {coin['name']}: {[d.strftime('%d-%m-%Y') for d in missing_dates]}")

                    price_data = [price for ts, price in data['prices']]
                    market_cap_data = [cap for ts, cap in data['market_caps']]
                    volume_data = [vol for ts, vol in data['total_volumes']]

                    # delete existing data
                    db.session.execute(
                        delete(CryptoMarketData).where(CryptoMarketData.name == coin['name'])
                    )
                    db.session.commit()

                    # Prepare list of new entries
                    new_entries = [
                        CryptoMarketData(
                            name=coin['name'],
                            symbol=coin['symbol'],
                            current_price=price,
                            market_cap=market_cap,
                            total_volume=volume,
                            date=date_,
                            image=f"assets/crypto/{coin['name']}-icon.png",
                        )
                        for date_, price, market_cap, volume in
                        zip(date_range_required, price_data, market_cap_data, volume_data)
                    ]

                    # Bulk insert
                    db.session.bulk_save_objects(new_entries)
                    db.session.commit()

                    logging.info(f"The price data for {coin['name']} has been updated with data from "
                                 f"{min(missing_dates)} to {max(missing_dates)}.")

            else:
                logging.info(f"The price data for {coin['name']} is up to date. No need to update anything.")


def crypto_db_to_df(coin_name):
    """
    Creates a pandas dataframe from SQL query of a crypto coin
    with the given symbol.
    """
    query = db.session.query(CryptoMarketData).filter(CryptoMarketData.name == coin_name)
    df = pd.read_sql(query.statement, db.engine)
    logging.info(f"Price data df to be processed:\n{df.head()}")

    if df.empty:
        logging.warning(f"No records found for symbol: {coin_name}")
    return df


def default_json_encoder(obj):
    """
    JSON serializer for non-serializable objects.
    Handles datetime, NumPy datetime64, and ndarray.
    """
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.datetime64):
        return pd.to_datetime(obj).isoformat()
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")


def get_crypto_options():
    """Returns a list of tuples with crypto id and name, extracted from the available crypto constants in the
    CryptoCoin class"""
    crypto_coins = [(coin.value["id"], coin.value["name"]) for coin in CryptoCoins]
    return crypto_coins


def save_charts_temp(chart_list):
    """
    Saves a list of chart data to a temporary JSON file tied to the user's session.

    Each session gets a unique file identified by a UUID stored in the session.
    Uses `default_json_encoder` to handle datetime objects in chart data.

    Args:
        chart_list (list): List of chart dictionaries to be saved.

    Returns:
        str: File path of the saved JSON file.
    """
    session_id = session.get("chart_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["chart_session_id"] = session_id

    file_path = os.path.join(SYS_TEMP_FOLDER, f"{session_id}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chart_list, f, default=default_json_encoder)

    return file_path


def load_charts_temp():
    """
    Loads the chart data JSON file associated with the user's session.

    If the session ID does not exist or the JSON file is missing/corrupted,
    returns None.

    Returns:
        list or None: List of chart dictionaries if successful, otherwise None.
    """
    session_id = session.get("chart_session_id")
    if not session_id:
        return None

    file_path = os.path.join(SYS_TEMP_FOLDER, f"{session_id}.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:

            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def clear_charts_temp():
    """
    Deletes the chart data JSON file associated with the user's session.

    If the session ID does not exist or the JSON file is missing/corrupted,
    returns None.

    Returns:
        list or None: List of chart dictionaries if successful, otherwise None.
    """
    session_id = session.get("chart_session_id")
    if not session_id:
        return None

    file_path = os.path.join(SYS_TEMP_FOLDER, f"{session_id}.json")

    try:
        os.remove(file_path)
        session["crypto_analysis_answer"] = None  # Clears AI answer from current session
        logging.info(f"Removed chart data JSON file: {session_id}.json")

    except FileNotFoundError:
        logging.info(f"Couldn't remove chart JSON, file not found: {session_id}.json")
        return None


def file_age_in_days(file_path):
    """
    Returns the age of a file in seconds based on its last modification time.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get the last modification time (in seconds since epoch)
    mod_time = os.path.getmtime(file_path)

    # Convert to datetime
    mod_datetime = dt.datetime.fromtimestamp(mod_time)

    # Calculate age
    age_days = (dt.datetime.now() - mod_datetime).seconds
    return age_days, mod_datetime


def clear_charts():
    """
    Clears chart temp files that are older than a specific timeframe.
    """
    maximum_file_age_in_seconds = 86400
    temp_file_list = os.listdir(SYS_TEMP_FOLDER)

    for filename in temp_file_list:
        file_path = os.path.join(SYS_TEMP_FOLDER, filename)
        file_age_data = file_age_in_days(file_path)
        file_age_in_seconds = file_age_data[0]
        file_last_modified = file_age_data[1]

        if os.path.isfile(file_path) and file_age_in_seconds > maximum_file_age_in_seconds:
            os.remove(file_path)
            logging.info(
                f"""Deleted {file_age_data}, 
                last modified: {file_last_modified}, 
                time since last modification: {file_age_in_seconds}""")

        else:
            logging.info(
                f"""Ignored {file_age_data}, 
                last modified: {file_last_modified}, 
                time since last modification: {file_age_in_seconds}""")


# Add job for clearing chart temps
scheduler.add_job(
    func=clear_charts,
    trigger=IntervalTrigger(days=1),
    id="clear_charts_temp",
    replace_existing=True
)


# Injects default variables to all templates
@app.context_processor
def inject_defaults():
    return dict(year=dt.datetime.now().year, enumerate=enumerate, zip=zip)


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")


# Function for the crypto tracker that uses an API to get the info and organize it
@app.route('/crypto-tracker', methods=["POST", "GET"])
def crypto_tracker():
    form = CryptoForm(crypto_options=get_crypto_options())

    dict_of_coins = {coin.value["id"]: coin.value["name"] for coin in CryptoCoins}
    history_range = app.config.get("HISTORY_RANGE", 30)

    page = int(request.args.get('page', 1))

    per_page = int(request.args.get('per_page', session.get('crypto_prices_per_page', 30)))
    session["crypto_prices_per_page"] = per_page

    if form.validate_on_submit():
        coin_id = form.crypto_select.data
        logging.info(f"Chosen coin_id: {coin_id}")
        session["crypto_coin_id"] = coin_id
    else:
        coin_id = session.get("crypto_coin_id", "bitcoin")
        form.crypto_select.data = coin_id
        logging.info(f"Displaying: {coin_id}")

    coin_name = dict_of_coins.get(coin_id)

    if not coin_id:
        flash("""Crypto tracker form didn't return a proper coin id. Please contact support 
            <a href="/contact-page">here</a>""", "main_error")
        logging.warning("Crypto tracker form didn't return a proper coin_id variable, the latter was a None type.")
        return redirect(url_for("crypto_tracker"))

    if not coin_name:
        flash("""Invalid coin name chosen. Please contact support 
                <a href="/contact-page">here</a>""", "main_error")
        logging.warning(f"Crypto tracker has a coin_id that isn't part of the recognized coin ids, "
                        f"no coin_name value was found for the provided coin_id: {coin_id}.")
        return redirect(url_for("crypto_tracker"))

    coin_in_db = (db.session.query(CryptoMarketData)
                  .filter(CryptoMarketData.name == coin_name)
                  .order_by(CryptoMarketData.date.desc())
                  .paginate(page=page, per_page=session.get('crypto_prices_per_page'), error_out=False))

    if not coin_in_db.items:  # `.items` is used for pagination results
        get_crypto_data(coin_name, coin_id)
        coin_in_db = (db.session.query(CryptoMarketData)
                      .filter(CryptoMarketData.name == coin_name)
                      .order_by(CryptoMarketData.date.desc())
                      .paginate(page=page, per_page=session.get('crypto_prices_per_page'), error_out=False))

    # Reverse to chronological order for X-axis
    items = list(reversed(coin_in_db.items))
    dates = [entry.date.strftime('%Y-%m-%d') for entry in items]
    prices = [entry.current_price for entry in items]

    fig = Figure()
    fig.add_trace(Scatter(x=dates, y=prices, mode='lines+markers', name='Price'))
    fig.update_layout(
        title=f'{coin_name} Price Over Time',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=400
    )

    chart_html = pio.to_html(fig, full_html=False)

    return render_template("crypto-tracker.html",
                           form=form,
                           coin_in_db=coin_in_db,
                           history_range=history_range,
                           coin_id=session.get("crypto_coin_id"),
                           chart_html=chart_html,
                           per_page=session.get('crypto_prices_per_page')
                           )


# AI crypto agent page
@app.route('/crypto-agent', methods=["POST", "GET"])
def crypto_agent():
    form = CryptoFormAI()
    min_char_length = 10

    if form.validate_on_submit():
        user_query = form.query.data
        logging.info(f"User query: {user_query}")

        # Avoid running the AI agent if the query is too short
        if len(user_query) < min_char_length:
            flash(f"The query has to be at least {min_char_length} characters long", "alert")
            return redirect(url_for("crypto_agent"))

        try:
            # Get AI answer and charts
            crypto_analysis_answer = crypto_ai_agent.stream_graph_updates(user_query=user_query)
            crypto_chart_list = crypto_ai_agent.get_chart_list()

            # Store answer in session
            session["crypto_analysis_answer"] = crypto_analysis_answer

            if crypto_chart_list:
                logging.info(f"Crypto chart list has {len(crypto_chart_list)} charts.")

                crypto_chart_list_json = []
                for chart in crypto_chart_list:
                    crypto_chart_list_json.append({
                        "chart_name": chart["chart_name"],
                        "plotly_json": chart["plotly_fig"].to_plotly_json(),
                        "interpretation": chart.get("interpretation", "")
                    })
            else:
                logging.info("Crypto chart list has no charts.")
                crypto_chart_list_json = []

            save_charts_temp(crypto_chart_list_json)

        except Exception as e:
            logging.error(f"❌ Error on crypto agent backend level: {e}")
            traceback.print_exc()
            flash(f"""An error occurred while handling your request: {e}. Please contact support 
            <a href="/contact-page">here</a>""", "main_error")

        finally:
            return redirect(url_for("crypto_agent"))

    # Convert Plotly JSON to HTML for template rendering
    chart_list = []
    loaded_charts = load_charts_temp()
    if loaded_charts:
        for chart in loaded_charts:
            chart_html = pio.to_html(
                chart["plotly_json"], full_html=False, include_plotlyjs='cdn'
            )

            chart_list.append({
                "chart_name": chart["chart_name"],
                "chart_html": chart_html,
                "interpretation": chart.get("interpretation", "")
                })

    return render_template(
        "crypto-agent.html",
        form=form,
        crypto_analysis_answer=session.get("crypto_analysis_answer"),
        crypto_chart_list=chart_list,
    )


# Helper Flask function to update crypto price data
@app.route('/update-price-data', methods=["POST", "GET"])
def update_price_data():
    update_crypto_data()
    source = request.args.get("source")
    flash("Prices were updated successfully!", "main_success")
    return redirect(url_for(source))


@app.route('/clear-ai-answer', methods=["POST", "GET"])
def clear_ai_answer():
    clear_charts_temp()
    source = request.args.get("source")
    flash("AI answer was deleted successfully!", "main_success")
    return redirect(url_for(source))


if __name__ == "__main__":
    # Ensure database exists
    with app.app_context():
        db.create_all()

        if not scheduler.running:
            scheduler.start()

    # Run Flask app
    app.run(debug=True)
