# predict_one.py
#
# Usage examples:
#   python predict_one.py "Patrick Mahomes" "BUF" "passing_yards" 275.5
#   python predict_one.py "Justin Jefferson" "GB" "receptions" 6.5
#
# Notes:
# - Uses model_final.keras + your nfl_player_gamelogs.json to compute features.
# - Requires the player_display_name to match exactly (case-sensitive).
# - If you prefer using player_id directly, see the comment in main().

import sys
import json
import numpy as np
import polars as pl
import tensorflow as tf

MODEL_PATH = "model_final.keras"
JSON_PATH  = "nfl_player_gamelogs.json"
TAU = 18.0  # must match training


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


def load_data(json_path: str) -> pl.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pl.DataFrame(data)

    df = df.with_columns(
        pl.col("game_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("game_date")
    ).sort(["player_id", "game_date"])

    return df


def get_player_id(df: pl.DataFrame, display_name: str) -> str:
    m = df.filter(pl.col("player_display_name") == display_name).select("player_id").unique()
    if m.height == 0:
        # Try a looser contains match to help debugging
        suggestions = (
            df.select("player_display_name")
              .unique()
              .filter(pl.col("player_display_name").str.contains(display_name, literal=False))
        )
        raise ValueError(
            f'Player "{display_name}" not found. '
            f"Try exact name from dataset. Suggestions:\n{suggestions.head(10)}"
        )
    return m[0, 0]


def predict_next_value(model, df: pl.DataFrame, player_id: str, opponent_team: str, stat_category: str):
    # Filter history for this player and stat
    if stat_category not in df.columns:
        raise ValueError(f"Unknown stat_category '{stat_category}'. Not a column in dataset.")

    player_games = df.filter(pl.col("player_id") == player_id).sort("game_date")
    if player_games.height < 2:
        raise ValueError("Not enough history: need at least 2 games to form features (uses prior games).")

    # Use ALL past games as history (up to the most recent game in data)
    hist_vals = player_games.select(stat_category).to_numpy().reshape(-1).astype(np.float32)
    # For predicting "next game", history is all games we have so far
    feats = build_features_from_history(hist_vals, TAU)  # shape (5,)

    inputs = {
        "player_id": tf.convert_to_tensor([player_id], dtype=tf.string),
        "opponent_team": tf.convert_to_tensor([opponent_team], dtype=tf.string),
        "stat_category": tf.convert_to_tensor([stat_category], dtype=tf.string),
        "num_feats": tf.convert_to_tensor([feats], dtype=tf.float32),  # shape (1,5)
    }

    pred = model.predict(inputs, verbose=0).reshape(-1)[0]
    return float(pred), feats


def main():
    if len(sys.argv) < 4:
        print('Usage: python test_model.py "Player Name" OPP_TEAM STAT_CATEGORY [LINE]')
        print('Example: python test_model.py "Patrick Mahomes" BUF passing_yards 275.5')
        sys.exit(1)

    player_name  = sys.argv[1]
    opponent     = sys.argv[2]
    stat_category = sys.argv[3]
    line = float(sys.argv[4]) if len(sys.argv) >= 5 else None

    model = tf.keras.models.load_model(MODEL_PATH)
    df = load_data(JSON_PATH)

    # If you prefer, replace this with a direct player_id input.
    player_id = get_player_id(df, player_name)

    pred, feats = predict_next_value(model, df, player_id, opponent, stat_category)

    print("---- Prediction ----")
    print("player:", player_name, "| player_id:", player_id)
    print("opponent:", opponent)
    print("stat_category:", stat_category)
    print("num_feats [ewm_mean, ewm_std, last1, mean5, hist_len]:", feats.tolist())
    print("predicted:", pred)

    if line is not None:
        decision = "OVER" if pred > line else "UNDER"
        print("line:", line, "=>", decision, "(pred - line =", pred - line, ")")

if __name__ == "__main__":
    main()
