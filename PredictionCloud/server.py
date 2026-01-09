# server.py
#
# FastAPI backend for Swift frontend.
#
# POST /predict
# Body:
# {
#   "player_name": "Patrick Mahomes",
#   "opponent": "BUF",
#   "stat_category": "passing_yards",
#   "line": 275.5   // optional
# }
#
# Response JSON includes:
# - prediction + num_feats
# - player_vs_opp_all_time_averages + games_count
# - player_vs_opp_last5: list of game objects
# - opp_def_last10_averages: allowed per game (pass/rush/total yards + TDs)
# - opp_def_last10_games: list of last-10 game totals (one per game)

from __future__ import annotations

import json
from typing import Optional, List, Dict
import numpy as np
import polars as pl
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "model_final.keras"
JSON_PATH  = "nfl_player_gamelogs.json"
TAU = 18.0

# stat columns present in your dataset
CATEGORIES = [
    "passing_yards", "completions", "attempts", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    "receiving_yards", "receptions", "receiving_tds",
]

QB_COLS = [
    "passing_yards", "completions", "attempts",
    "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
]
WRTE_COLS = ["receiving_yards", "receptions", "receiving_tds"]
RB_COLS = ["rushing_yards", "rushing_tds", "receiving_yards", "receptions", "receiving_tds"]

ID_COLS = ["game_date", "season", "week", "season_type", "team", "opponent_team"]


# ----------------------------
# API models (Swift-friendly)
# ----------------------------
class PredictRequest(BaseModel):
    player_name: str
    opponent: str
    stat_category: str
    line: Optional[float] = None


class PlayerVsOpponentAllTime(BaseModel):
    games: int
    averages: Dict[str, float]


class OppDefenseLast10Averages(BaseModel):
    games: int
    pass_yards_allowed: float
    rush_yards_allowed: float
    total_yards_allowed: float
    pass_tds_allowed: float
    rush_tds_allowed: float
    total_tds_allowed: float


class PredictResponse(BaseModel):
    player_id: str
    player_name: str
    opponent: str
    stat_category: str
    num_feats: List[float]
    prediction: float
    line: Optional[float] = None
    decision: Optional[str] = None
    edge: Optional[float] = None

    player_vs_opp_all_time: PlayerVsOpponentAllTime
    player_vs_opp_last5: List[Dict[str, object]]

    opp_def_last10_averages: Optional[OppDefenseLast10Averages] = None
    opp_def_last10_games: List[Dict[str, object]]


app = FastAPI(title="NFL Props Predictor API")

model: tf.keras.Model | None = None
df_all: pl.DataFrame | None = None


# ----------------------------
# Helpers
# ----------------------------
def exp_weights_all(n: int, tau: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float32)
    w = np.exp(-(n - 1 - idx) / tau)
    w = w / (w.sum() + 1e-8)
    return w


def build_features_from_history(values: np.ndarray, tau: float) -> np.ndarray:
    hist = np.nan_to_num(values.astype(np.float32), nan=0.0)
    n = len(hist)

    w = exp_weights_all(n, tau)
    mu = float((hist * w).sum())
    var = float(((hist - mu) ** 2 * w).sum())
    sd = float(np.sqrt(var))

    last1 = float(hist[-1])
    last5 = hist[-5:] if n >= 5 else hist
    mean5 = float(last5.mean()) if n else 0.0

    return np.array([mu, sd, last1, mean5, float(n)], dtype=np.float32)


def choose_relevant_stat_cols(position_group: str, stat_category: str, available_cols: set[str]) -> List[str]:
    if position_group == "QB":
        cols = QB_COLS[:]
    elif position_group == "RB":
        cols = RB_COLS[:]
    elif position_group in ("WR", "TE"):
        cols = WRTE_COLS[:]
    else:
        cols = [stat_category]

    if stat_category not in cols:
        cols = [stat_category] + cols

    # only keep those that exist in df
    return [c for c in cols if c in available_cols]


def to_records(df: pl.DataFrame) -> List[Dict[str, object]]:
    recs = df.to_dicts()
    # make dates JSON-safe
    for r in recs:
        if "game_date" in r and r["game_date"] is not None:
            r["game_date"] = str(r["game_date"])
    return recs


def get_player_id_or_none(df: pl.DataFrame, display_name: str) -> Optional[str]:
    m = df.filter(pl.col("player_display_name") == display_name).select("player_id").unique()
    if m.height == 0:
        return None
    return m[0, 0]


def get_suggestions(df: pl.DataFrame, query: str) -> List[str]:
    s = (
        df.select("player_display_name")
          .unique()
          .filter(pl.col("player_display_name").str.contains(query, literal=False))
          .head(10)
    )
    return s["player_display_name"].to_list() if s.height > 0 else []


def col_or_zero(name: str, available_cols: set[str]):
    return pl.col(name) if name in available_cols else pl.lit(0)


# ----------------------------
# Startup (load once)
# ----------------------------
@app.on_event("startup")
def startup():
    global model, df_all
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    df_all = pl.DataFrame(data)

    df_all = (
        df_all.with_columns(
            pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("game_date")
        )
        .sort(["player_id", "game_date"])
    )


# ----------------------------
# Core computations
# ----------------------------
def predict_value(player_id: str, opponent: str, stat_category: str):
    assert df_all is not None and model is not None

    player_games = df_all.filter(pl.col("player_id") == player_id).sort("game_date")
    if player_games.height < 2:
        raise HTTPException(status_code=400, detail="Not enough history for this player.")

    hist_vals = player_games.select(stat_category).to_numpy().reshape(-1).astype(np.float32)
    feats = build_features_from_history(hist_vals, TAU)

    inputs = {
        "player_id": tf.convert_to_tensor([player_id], dtype=tf.string),
        "opponent_team": tf.convert_to_tensor([opponent], dtype=tf.string),
        "stat_category": tf.convert_to_tensor([stat_category], dtype=tf.string),
        "num_feats": tf.convert_to_tensor([feats], dtype=tf.float32),
    }
    pred = float(model.predict(inputs, verbose=0).reshape(-1)[0])
    return pred, feats.tolist(), player_games


def player_vs_opponent_stats(player_games: pl.DataFrame, opponent: str, stat_category: str):
    available = set(player_games.columns)

    pos_group = "UNKNOWN"
    if "position_group" in available and player_games.height > 0:
        pos_group = player_games.select("position_group")[0, 0]

    stat_cols = choose_relevant_stat_cols(pos_group, stat_category, available)

    vs = player_games.filter(pl.col("opponent_team") == opponent).sort("game_date")
    if vs.height == 0:
        return PlayerVsOpponentAllTime(games=0, averages={}), []

    # All-time averages
    avg_row = (
        vs.select([pl.col(c).mean().alias(c) for c in stat_cols])
          .row(0, named=True)
    )
    averages = {c: float(avg_row[c]) for c in stat_cols if avg_row.get(c) is not None}

    # Last 5 per-game
    cols_for_last5 = [c for c in (ID_COLS + stat_cols) if c in vs.columns]
    last5 = to_records(vs.tail(5).select(cols_for_last5))

    return PlayerVsOpponentAllTime(games=vs.height, averages=averages), last5


def opponent_def_allowed_last10(opponent: str):
    """
    Approx defensive allowed per game for opponent over their last 10 games.
    Uses your dataset:
      - filter rows with opponent_team == opponent
      - group into games by (game_date, season, week, season_type, offense team)
      - sum offense totals: pass/rush yards + pass/rush TDs
    """
    assert df_all is not None
    available = set(df_all.columns)

    required = ["game_date", "season", "week", "season_type", "team", "opponent_team"]
    for k in required:
        if k not in available:
            return None, []

    against = df_all.filter(pl.col("opponent_team") == opponent)
    if against.height == 0:
        return None, []

    games = (
        against
        .group_by(["game_date", "season", "week", "season_type", "team"], maintain_order=True)
        .agg([
            col_or_zero("passing_yards", available).sum().alias("pass_yards_allowed"),
            col_or_zero("rushing_yards", available).sum().alias("rush_yards_allowed"),
            col_or_zero("passing_tds", available).sum().alias("pass_tds_allowed"),
            col_or_zero("rushing_tds", available).sum().alias("rush_tds_allowed"),
        ])
        .with_columns([
            (pl.col("pass_yards_allowed") + pl.col("rush_yards_allowed")).alias("total_yards_allowed"),
            (pl.col("pass_tds_allowed") + pl.col("rush_tds_allowed")).alias("total_tds_allowed"),
        ])
        .sort("game_date")
    )

    last10 = games.tail(10)
    avg = last10.select([
        pl.col("pass_yards_allowed").mean().alias("pass_yards_allowed"),
        pl.col("rush_yards_allowed").mean().alias("rush_yards_allowed"),
        pl.col("total_yards_allowed").mean().alias("total_yards_allowed"),
        pl.col("pass_tds_allowed").mean().alias("pass_tds_allowed"),
        pl.col("rush_tds_allowed").mean().alias("rush_tds_allowed"),
        pl.col("total_tds_allowed").mean().alias("total_tds_allowed"),
    ]).row(0, named=True)

    averages = OppDefenseLast10Averages(
        games=last10.height,
        pass_yards_allowed=float(avg["pass_yards_allowed"]),
        rush_yards_allowed=float(avg["rush_yards_allowed"]),
        total_yards_allowed=float(avg["total_yards_allowed"]),
        pass_tds_allowed=float(avg["pass_tds_allowed"]),
        rush_tds_allowed=float(avg["rush_tds_allowed"]),
        total_tds_allowed=float(avg["total_tds_allowed"]),
    )

    last10_records = to_records(
        last10.select([
            "game_date", "season", "week", "season_type", "team",
            "pass_yards_allowed", "rush_yards_allowed", "total_yards_allowed",
            "pass_tds_allowed", "rush_tds_allowed", "total_tds_allowed",
        ])
    )

    return averages, last10_records


# ----------------------------
# Endpoint
# ----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    assert df_all is not None

    if req.stat_category not in set(CATEGORIES):
        raise HTTPException(status_code=400, detail=f"Unknown stat_category '{req.stat_category}'")

    player_id = get_player_id_or_none(df_all, req.player_name)
    if player_id is None:
        suggestions = get_suggestions(df_all, req.player_name)
        raise HTTPException(
            status_code=404,
            detail={"message": f'Player "{req.player_name}" not found.', "suggestions": suggestions},
        )

    pred, feats, player_games = predict_value(player_id, req.opponent, req.stat_category)

    decision = None
    edge = None
    if req.line is not None:
        edge = pred - float(req.line)
        decision = "OVER" if edge > 0 else "UNDER"

    p_all_time, p_last5 = player_vs_opponent_stats(player_games, req.opponent, req.stat_category)
    opp_avg, opp_last10 = opponent_def_allowed_last10(req.opponent)

    return PredictResponse(
        player_id=player_id,
        player_name=req.player_name,
        opponent=req.opponent,
        stat_category=req.stat_category,
        num_feats=[float(x) for x in feats],
        prediction=float(pred),
        line=req.line,
        decision=decision,
        edge=edge,
        player_vs_opp_all_time=p_all_time,
        player_vs_opp_last5=p_last5,
        opp_def_last10_averages=opp_avg,
        opp_def_last10_games=opp_last10,
    )
