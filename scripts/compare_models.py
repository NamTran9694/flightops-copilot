# --- A. Imports and settings
"""Compare forecasting models with time-series back-testing.

Models: naive, seasonal naive, linear (production), XGBoost, hybrid (linear + XGBoost).
Run:  python scripts/compare_models.py
Needs: pip install -r requirements-ml.txt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")                        # draw charts to files (no screen needed)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from flightops.forecast import MODEL_CONFIG, SeasonalTrendModel, load_daily, mape  # noqa: E402

REPORTS = ROOT / "reports"
METRICS = ["flights", "hours_per_flight", "labor_hours"]
TEST_DAYS = 60                               # size of each test window
FOLDS = 3                                    # number of back-test windows

# --- B. The 5 models
# ---------- A. Calendar features for XGBoost ----------
def calendar_features(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "t": (dates - origin).days,          # days since start (trend)
        "dow": dates.dayofweek,              # 0 = Monday ... 6 = Sunday
        "month": dates.month,
        "dayofyear": dates.dayofyear,
    }, index=dates)


# ---------- B. The 5 models: each one learns from `train` and predicts `dates` ----------
def naive(train: pd.Series, dates: pd.DatetimeIndex, metric: str) -> np.ndarray:
    """Average of the last 30 days."""
    return np.full(len(dates), train.iloc[-30:].mean())


def seasonal_naive(train: pd.Series, dates: pd.DatetimeIndex, metric: str) -> np.ndarray:
    """Repeat the last week: next Monday = last Monday, and so on."""
    last_week = train.iloc[-7:]
    by_weekday = dict(zip(last_week.index.dayofweek, last_week.values))
    return np.array([by_weekday[d] for d in dates.dayofweek])


def linear(train: pd.Series, dates: pd.DatetimeIndex, metric: str) -> np.ndarray:
    """Our production model, with the same settings as forecast.py."""
    cfg = MODEL_CONFIG[metric]
    y = train.iloc[-cfg["history_days"]:] if cfg["history_days"] else train
    return SeasonalTrendModel(months=cfg["months"]).fit(y).predict(dates)


def xgboost_model(train: pd.Series, dates: pd.DatetimeIndex, metric: str) -> np.ndarray:
    """Gradient-boosted trees on calendar features."""
    origin = train.index[0]
    model = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.9, random_state=42)
    model.fit(calendar_features(train.index, origin), train.values)
    return model.predict(calendar_features(dates, origin))


def hybrid(train: pd.Series, dates: pd.DatetimeIndex, metric: str) -> np.ndarray:
    """Linear model for trend + seasonality, XGBoost learns what linear misses."""
    origin = train.index[0]
    base_train = linear(train, train.index, metric)
    residuals = train.values - base_train
    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(calendar_features(train.index, origin), residuals)
    return linear(train, dates, metric) + model.predict(calendar_features(dates, origin))


MODELS = {"naive": naive, "seasonal_naive": seasonal_naive, "linear": linear,
          "xgboost": xgboost_model, "hybrid": hybrid}

# --- C. The rolling back-test

def backtest(y: pd.Series, metric: str) -> dict:
    """Test on 3 different 60-day windows; return the average MAPE per model."""
    scores = {name: [] for name in MODELS}
    for fold in range(FOLDS):
        end = len(y) - fold * TEST_DAYS
        train, test = y.iloc[:end - TEST_DAYS], y.iloc[end - TEST_DAYS:end]
        for name, fn in MODELS.items():
            scores[name].append(mape(test.values, fn(train, test.index, metric)))
    return {name: round(float(np.mean(s)), 2) for name, s in scores.items()}

# --- D. Leaderboard and chart
def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load_daily()

    # 1. Leaderboard
    rows = {m: backtest(df[m].astype(float), m) for m in METRICS}
    board = pd.DataFrame(rows)                       # models x metrics
    board.index.name = "model"
    print("\nBack-test MAPE % (average of 3 windows of 60 days, lower is better)\n")
    print(board.to_string())
    print("\nChampion per metric:")
    for m in METRICS:
        print(f"  {m:<17} -> {board[m].idxmin()} ({board[m].min()}%)")
    board.to_csv(REPORTS / "model_comparison.csv")

    # 2. Chart: one-year-ahead flights forecast, linear vs XGBoost
    y = df["flights"].astype(float)
    future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=365, freq="D")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(y.index, y.rolling(7).mean(), color="#555555", label="Actual (7-day average)")
    for name, color in [("linear", "#1f77b4"), ("xgboost", "#d62728")]:
        pred = pd.Series(MODELS[name](y, future, "flights"), index=future)
        ax.plot(future, pred.rolling(7).mean(), color=color, label=f"{name} forecast")
    ax.set_title("Flights per day: 1-year forecast. XGBoost cannot extrapolate the growth trend")
    ax.set_ylabel("Flights per day")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "forecast_comparison.png", dpi=120)
    print(f"\nSaved {REPORTS / 'model_comparison.csv'} and {REPORTS / 'forecast_comparison.png'}")


if __name__ == "__main__":
    main()