# predict_one.py
#
# Usage examples:
#   python predict_one.py "Patrick Mahomes" "BUF" "passing_yards" 275.5
#   python predict_one.py "Justin Jefferson" "GB" "receptions" 6.5
#
# What it does:
# - Loads model_final.keras + your nfl_player_gamelogs.json
# - Computes exp-decay "recent form" features from ALL prior games for (player, stat_category)
# - Predicts the next-game stat value vs the opponent
# - ALSO prints the player's per-game stat log vs that opponent:
#     (a) all-time vs that opponent
#     (b) last 5 games vs that opponent (if available; no "less than 5" message)

import sys
import json
import numpy as np
import polars as pl
import tensorflow as tf

MODEL_PATH = "model_final.keras"
JSON_PATH  = "nfl_player_gamelogs.json"
TAU = 18.0  # must match training


# ----------------------------
# Decay helpers (must match training)
# ----------------------------
def exp_weights_all(n: int, tau: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float32)
    w = np.exp(-(n - 1 - idx) / tau)
    w = w / (w.sum() + 1e-8)
    return w


def build_features_from_history(values: np.ndarray, tau: float) -> np.ndarray:
    """
    values: prior game values (oldest->newest), length n>=1
    returns num_feats: [ewm_mean, ewm_std, last1, mean5, hist_len]
    """
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


# ----------------------------
# Data loading
# ----------------------------
def load_data(json_path: str) -> pl.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pl.DataFrame(data)

    df = (
        df.with_columns(
            pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("game_date")
        )
        .sort(["player_id", "game_date"])
    )
    return df


# ----------------------------
# Player lookup
# ----------------------------
def get_player_id_or_none(df: pl.DataFrame, display_name: str):
    m = df.filter(pl.col("player_display_name") == display_name).select("player_id").unique()
    if m.height == 0:
        suggestions = (
            df.select("player_display_name")
              .unique()
              .filter(pl.col("player_display_name").str.contains(display_name, literal=False))
              .head(10)
        )
        return None, suggestions
    return m[0, 0], None


# ----------------------------
# Relevant columns to print vs opponent
# ----------------------------
QB_COLS = [
    "passing_yards", "completions", "attempts",
    "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
]
WRTE_COLS = ["receiving_yards", "receptions", "receiving_tds"]
RB_COLS = ["rushing_yards", "rushing_tds", "receiving_yards", "receptions", "receiving_tds"]

ID_COLS = ["game_date", "season", "week", "season_type", "team", "opponent_team"]


def choose_relevant_cols(df: pl.DataFrame, position_group: str, stat_category: str):
    """
    Choose which stats to print for per-game logs.
    Preference:
      - If position_group suggests QB/RB/WR/TE, use that set.
      - Always include the requested stat_category if present.
      - Include only columns that exist.
    """
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

    cols = [c for c in cols if c in df.columns]
    return ID_COLS + cols


# ----------------------------
# Prediction
# ----------------------------
def predict_next_value(model, df: pl.DataFrame, player_id: str, opponent_team: str, stat_category: str):
    if stat_category not in df.columns:
        raise ValueError(f"Unknown stat_category '{stat_category}'. Not a column in dataset.")

    player_games = df.filter(pl.col("player_id") == player_id).sort("game_date")
    if player_games.height < 2:
        raise ValueError("Not enough history: need at least 2 games to form features (uses prior games).")

    hist_vals = player_games.select(stat_category).to_numpy().reshape(-1).astype(np.float32)
    feats = build_features_from_history(hist_vals, TAU)

    inputs = {
        "player_id": tf.convert_to_tensor([player_id], dtype=tf.string),
        "opponent_team": tf.convert_to_tensor([opponent_team], dtype=tf.string),
        "stat_category": tf.convert_to_tensor([stat_category], dtype=tf.string),
        "num_feats": tf.convert_to_tensor([feats], dtype=tf.float32),
    }

    pred = model.predict(inputs, verbose=0).reshape(-1)[0]
    return float(pred), feats, player_games


# ----------------------------
# Printing opponent logs
# ----------------------------
def print_vs_opponent_logs(player_games: pl.DataFrame, opponent_team: str, stat_category: str):
    # Determine position_group for relevant stats selection
    pos_group = None
    if "position_group" in player_games.columns and player_games.height > 0:
        pos_group = player_games.select("position_group")[0, 0]
    else:
        pos_group = "UNKNOWN"

    cols_to_show = choose_relevant_cols(player_games, pos_group, stat_category)

    vs = (
        player_games
        .filter(pl.col("opponent_team") == opponent_team)
        .sort("game_date")
    )

    if vs.height == 0:
        print(f"\nNo games found vs {opponent_team} in this dataset.")
        return

    vs_all = vs.select([c for c in cols_to_show if c in vs.columns])
    print(f"\n---- All-time vs {opponent_team} (per-game) ----")
    # print in a readable multi-line table format
    print(vs_all)

    vs_last5 = vs.tail(5).select([c for c in cols_to_show if c in vs.columns])
    print(f"\n---- Last 5 vs {opponent_team} ----")
    print(vs_last5)


# ----------------------------
# Main
# ----------------------------
def main():
    if len(sys.argv) < 4:
        print('Usage: python predict_one.py "Player Name" OPP_TEAM STAT_CATEGORY [LINE]')
        print('Example: python predict_one.py "Patrick Mahomes" BUF passing_yards 275.5')
        sys.exit(1)

    player_name   = sys.argv[1]
    opponent      = sys.argv[2]
    stat_category = sys.argv[3]
    line = float(sys.argv[4]) if len(sys.argv) >= 5 else None

    model = tf.keras.models.load_model(MODEL_PATH)
    df = load_data(JSON_PATH)

    player_id, suggestions = get_player_id_or_none(df, player_name)
    if player_id is None:
        print(f'Player "{player_name}" not found in dataset.')
        if suggestions is not None and suggestions.height > 0:
            print("Closest matches:")
            print(suggestions)
        return

    pred, feats, player_games = predict_next_value(model, df, player_id, opponent, stat_category)

    print("---- Prediction ----")
    print("player:", player_name, "| player_id:", player_id)
    print("opponent:", opponent)
    print("stat_category:", stat_category)
    print("num_feats [ewm_mean, ewm_std, last1, mean5, hist_len]:", feats.tolist())
    print("predicted:", pred)

    if line is not None:
        decision = "OVER" if pred > line else "UNDER"
        print("line:", line, "=>", decision, "(pred - line =", pred - line, ")")

    # Print historical logs vs opponent (all-time + last 5)
    print_vs_opponent_logs(player_games, opponent, stat_category)


if __name__ == "__main__":
    main()
