import json
import os
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load the saved model ONCE, when the server starts
model = joblib.load("model.joblib")
with open("model_info.json") as f:
    info = json.load(f)

app = FastAPI(
    title="Stock price prediction API",
    description="Predicts the next closing price from the last two closes. "
    "A teaching example, not investment advice.",
)

# Allow web pages on other addresses (any frontend) to call this API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

# Closes only change once a day, so there is no reason to hit the provider
# on every request. Keyed by symbol -> (fetched_at, quote).
CACHE_TTL = 900  # seconds
_cache: dict[str, tuple[float, dict]] = {}


def predict_next(lag_1: float, lag_2: float) -> float:
    X = pd.DataFrame({"lag_1": [lag_1], "lag_2": [lag_2]})
    return round(float(model.predict(X)[0]), 2)


def _fetch_finnhub(symbol: str, key: str) -> dict:
    """Finnhub /quote: 60 calls/minute on the free tier, no daily cap."""
    reply = requests.get(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": symbol, "token": key},
        timeout=15,
    ).json()

    if not reply.get("c"):
        raise HTTPException(
            status_code=502,
            detail=f"No quote found for symbol '{symbol}'. Provider said: {reply}",
        )

    day = None
    if reply.get("t"):
        day = datetime.fromtimestamp(reply["t"], tz=timezone.utc).date().isoformat()

    return {
        "symbol": symbol,
        "latest_trading_day": day,
        "last_close": float(reply["c"]),
        "previous_close": float(reply["pc"]),
        "source": "finnhub",
    }


def _fetch_alphavantage(symbol: str, key: str) -> dict:
    """Fallback. Free tier is 25 requests/day AND rate limited per IP, so this
    works from a laptop but usually fails from a shared cloud host."""
    reply = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key},
        timeout=15,
    ).json()

    quote = reply.get("Global Quote")
    if not quote:
        raise HTTPException(
            status_code=502,
            detail=f"No quote found for symbol '{symbol}'. Provider said: {reply}",
        )

    return {
        "symbol": quote["01. symbol"],
        "latest_trading_day": quote["07. latest trading day"],
        "last_close": float(quote["05. price"]),
        "previous_close": float(quote["08. previous close"]),
        "source": "alphavantage",
    }


def fetch_quote(symbol: str) -> dict:
    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol must not be empty.")

    now = time.time()
    cached = _cache.get(symbol)
    if cached and now - cached[0] < CACHE_TTL:
        return {**cached[1], "cached": True}

    finnhub_key = os.environ.get("FINNHUB_KEY")
    alpha_key = os.environ.get("ALPHAVANTAGE_KEY")

    if finnhub_key:
        quote = _fetch_finnhub(symbol, finnhub_key)
    elif alpha_key:
        quote = _fetch_alphavantage(symbol, alpha_key)
    else:
        raise HTTPException(
            status_code=500,
            detail="No data provider configured. Set FINNHUB_KEY (preferred) "
            "or ALPHAVANTAGE_KEY on the server.",
        )

    _cache[symbol] = (now, quote)
    return {**quote, "cached": False}


@app.get("/")
def home():
    return {"message": "The API is running. Open /docs to try it.", **info}


@app.get("/predict")
def predict(lag_1: float, lag_2: float):
    """Predict tomorrow's close. lag_1 = latest close, lag_2 = the close before it."""
    return {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "predicted_next_close": predict_next(lag_1, lag_2),
        "naive_forecast": lag_1,
    }


@app.get("/predict/live")
def predict_live(symbol: str = "TSLA"):
    """Get the latest two closes from the data provider, then predict the next one."""
    quote = fetch_quote(symbol)
    return {
        **quote,
        "predicted_next_close": predict_next(quote["last_close"], quote["previous_close"]),
        "naive_forecast": quote["last_close"],
        "model": info["model"],
        "trained_on": info["trained_on"],
    }
