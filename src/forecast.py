import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def _daily_series(orders, value_column):
    data = orders.copy()
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    daily = data.groupby("order_date")[value_column].sum().rename("value").to_frame()
    date_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(date_index, fill_value=0).rename_axis("date").reset_index()


def forecast_daily(orders, value_column="realized_sales_amount", horizon=30):
    """Trend ve haftanın günü etkisiyle basit günlük tahmin üretir."""
    history = _daily_series(orders, value_column)
    history["trend"] = np.arange(len(history))
    history["dow"] = history["date"].dt.dayofweek
    history["dow_sin"] = np.sin(2 * np.pi * history["dow"] / 7)
    history["dow_cos"] = np.cos(2 * np.pi * history["dow"] / 7)

    features = ["trend", "dow_sin", "dow_cos"]
    model = LinearRegression()
    model.fit(history[features], history["value"])
    history["kind"] = "Gerçekleşen"

    future_dates = pd.date_range(
        history["date"].max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )
    future = pd.DataFrame({"date": future_dates})
    future["trend"] = np.arange(len(history), len(history) + horizon)
    future["dow"] = future["date"].dt.dayofweek
    future["dow_sin"] = np.sin(2 * np.pi * future["dow"] / 7)
    future["dow_cos"] = np.cos(2 * np.pi * future["dow"] / 7)
    future["value"] = model.predict(future[features]).clip(min=0)
    future["kind"] = "Tahmin"

    return pd.concat(
        [history[["date", "value", "kind"]], future[["date", "value", "kind"]]],
        ignore_index=True,
    )


def backtest_daily(orders, value_column="realized_sales_amount", horizon=30):
    """Son dönemi holdout kabul ederek basit forecast hatasını ölçer."""
    history = _daily_series(orders, value_column)

    if len(history) <= horizon * 2:
        return {
            "horizon": horizon,
            "mae": None,
            "wape": None,
        }

    train = history.iloc[:-horizon].copy()
    test = history.iloc[-horizon:].copy()

    for frame in [train, test]:
        frame["trend"] = np.arange(len(frame))
        frame["dow"] = frame["date"].dt.dayofweek
        frame["dow_sin"] = np.sin(2 * np.pi * frame["dow"] / 7)
        frame["dow_cos"] = np.cos(2 * np.pi * frame["dow"] / 7)

    # Test döneminin trend index'i train'in devamı olmalı.
    test["trend"] = np.arange(len(train), len(train) + len(test))
    features = ["trend", "dow_sin", "dow_cos"]

    model = LinearRegression()
    model.fit(train[features], train["value"])
    prediction = model.predict(test[features]).clip(min=0)

    mae = mean_absolute_error(test["value"], prediction)
    denominator = test["value"].sum()
    wape = (abs(test["value"].to_numpy() - prediction).sum() / denominator * 100) if denominator else None

    return {
        "horizon": horizon,
        "mae": float(mae),
        "wape": float(wape) if wape is not None else None,
    }
