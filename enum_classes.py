from enum import Enum


class CryptoValues(Enum):
    """Enum class of values related to the handling of crypto tracker"""
    HISTORY_RANGE = (365, "Maximum number of past days available for fetching historical cryptocurrency price data.")
    NEWS_RANGE = (30, "Maximum number of past days allowed for retrieving news articles related to a cryptocurrency.")

    def __init__(self, value, description):
        self._value_ = value
        self.description = description


class CryptoCoins(Enum):
    """Enum class of dicts with the name, id and symbol of select crypto coins"""
    BITCOIN = {"name": "Bitcoin", "id": "bitcoin", "symbol": "BTC"}
    ETHEREUM = {"name": "Ethereum", "id": "ethereum", "symbol": "ETH"}
    DOGECOIN = {"name": "Dogecoin", "id": "dogecoin", "symbol": "DOGE"}
    AVALANCHE = {"name": "Avalanche", "id": "avalanche-2", "symbol": "AVAX"}
    TETHER = {"name": "Tether", "id": "tether", "symbol": "USDT"}