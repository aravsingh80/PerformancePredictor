# predict_one.py
#
# Usage examples:
#   python predict_one.py "Patrick Mahomes" "BUF" "passing_yards" 275.5
#   python predict_one.py "Justin Jefferson" "GB" "receptions" 6.5
#
# Notes:
# - Uses model_final.keras + your nfl_player_gamelogs.json to compute features.
# - Requires the player_display_name to match exactly (case-sensitive).
# - Outputs:
#   1) Model prediction
#   2) All-time vs opponent: AVERAGES (plus games count)
#   3) Last 5 vs opponent: per-game pretty JSON blocks
#   4) Opponent defense (approx): yards/TDs allowed per game over their last 10 games
#
# IMPORTANT:
# The "opponent allowed" section is computed from your player-game-log dataset by:
# - filtering all player rows where opponent_team == OPP
# - grouping into games (by date/season/week/season_type/offense team)
# - summing offense totals: passing_yards + rushing_yards, passing_tds + rushing_tds
# This is a good approximation using your available data (no schedules/defense table needed).

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
# Relevant columns to show vs opponent
# ----------------------------
QB_COLS = [
    "passing_yards", "completions", "attempts",
    "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
]
WRTE_COLS = ["receiving_yards", "receptions", "receiving_tds"]
RB_COLS = ["rushing_yards", "rushing_tds", "receiving_yards", "receptions", "receiving_tds"]

ID_COLS = ["game_date", "season", "week", "season_type", "team", "opponent_team"]


def choose_relevant_stat_cols(df: pl.DataFrame, position_group: str, stat_category: str) -> list[str]:
    """
    Returns stat columns (NOT id columns) relevant to the player's position group.
    Ensures requested stat_category is included (if present).
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

    return [c for c in cols if c in df.columns]


def pl_date_to_str(x):
    try:
        return x.isoformat()
    except Exception:
        return str(x)


def df_to_pretty_records(df: pl.DataFrame) -> list[dict]:
    recs = df.to_dicts()
    for r in recs:
        if "game_date" in r and r["game_date"] is not None:
            r["game_date"] = pl_date_to_str(r["game_date"])
    return recs


def print_last_games_json(title: str, df: pl.DataFrame):
    recs = df_to_pretty_records(df)
    print(f"\n{title} (games: {len(recs)})")
    for i, r in enumerate(recs, 1):
        print(f"\nGame {i}:")
        print(json.dumps(r, indent=2))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


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
# Opponent summary + last 5 logs (player vs opponent)
# ----------------------------
def print_vs_opponent_summary_and_last5(player_games: pl.DataFrame, opponent_team: str, stat_category: str):
    if player_games.height == 0:
        print("\nNo games for this player in dataset.")
        return

    pos_group = "UNKNOWN"
    if "position_group" in player_games.columns:
        pos_group = player_games.select("position_group")[0, 0]

    stat_cols = choose_relevant_stat_cols(player_games, pos_group, stat_category)

    vs = (
        player_games
        .filter(pl.col("opponent_team") == opponent_team)
        .sort("game_date")
    )

    if vs.height == 0:
        print(f"\nNo games found vs {opponent_team} in this dataset.")
        return

    # 1) ALL-TIME AVERAGES (player vs opponent)
    agg_exprs = [pl.col(c).mean().alias(c) for c in stat_cols if c in vs.columns]
    avg_df = vs.select(agg_exprs) if agg_exprs else pl.DataFrame()

    print(f"\n---- All-time vs {opponent_team} Averages ----")
    print(f"games: {vs.height}")
    if avg_df.height > 0:
        avg_row = avg_df.row(0, named=True)
        for k in stat_cols:
            if k in avg_row:
                v = safe_float(avg_row[k])
                if v is None:
                    continue
                print(f"{k}: {v:.3f}")
    else:
        print("(No relevant stat columns available to average.)")

    # 2) LAST 5 PER-GAME (player vs opponent)
    cols_for_last5 = [c for c in (ID_COLS + stat_cols) if c in vs.columns]
    vs_last5 = vs.tail(5).select(cols_for_last5)

    print_last_games_json(f"---- Last 5 vs {opponent_team} ----", vs_last5)


# ----------------------------
# NEW: Opponent defense allowed (last 10 games)
# ----------------------------
def print_opponent_allowed_last10(df_all: pl.DataFrame, opponent_team: str):
    """
    Approx opponent defensive allowed per game over last 10 games:
    - Filter all player rows where opponent_team == opponent_team
    - Group into games by (game_date, season, week, season_type, offense_team)
    - Sum offense totals:
        total_yards = sum(passing_yards) + sum(rushing_yards)
        total_tds   = sum(passing_tds) + sum(rushing_tds)
    """
    required_keys = ["game_date", "season", "week", "season_type", "team", "opponent_team"]
    for k in required_keys:
        if k not in df_all.columns:
            print(f"\n---- {opponent_team} Defense (last 10) ----")
            print(f"Cannot compute opponent-allowed stats; missing column: {k}")
            return

    # Make sure these stat columns exist; if not, treat as 0
    def col_or_zero(name: str):
        return pl.col(name) if name in df_all.columns else pl.lit(0)

    against_opp = (
        df_all
        .filter(pl.col("opponent_team") == opponent_team)
        .select(required_keys + [c for c in ["passing_yards", "rushing_yards", "passing_tds", "rushing_tds"] if c in df_all.columns])
    )

    if against_opp.height == 0:
        print(f"\n---- {opponent_team} Defense (last 10) ----")
        print("No games found for this opponent in dataset.")
        return

    # Build per-game totals allowed by this opponent
    games = (
        against_opp
        .group_by(["game_date", "season", "week", "season_type", "team"], maintain_order=True)
        .agg([
            col_or_zero("passing_yards").sum().alias("pass_yards_allowed"),
            col_or_zero("rushing_yards").sum().alias("rush_yards_allowed"),
            col_or_zero("passing_tds").sum().alias("pass_tds_allowed"),
            col_or_zero("rushing_tds").sum().alias("rush_tds_allowed"),
        ])
        .with_columns([
            (pl.col("pass_yards_allowed") + pl.col("rush_yards_allowed")).alias("total_yards_allowed"),
            (pl.col("pass_tds_allowed") + pl.col("rush_tds_allowed")).alias("total_tds_allowed"),
        ])
        .sort("game_date")
    )

    last10 = games.tail(10)

    # Averages over last 10 games
    avg = last10.select([
        pl.col("pass_yards_allowed").mean().alias("avg_pass_yards_allowed"),
        pl.col("rush_yards_allowed").mean().alias("avg_rush_yards_allowed"),
        pl.col("total_yards_allowed").mean().alias("avg_total_yards_allowed"),
        pl.col("pass_tds_allowed").mean().alias("avg_pass_tds_allowed"),
        pl.col("rush_tds_allowed").mean().alias("avg_rush_tds_allowed"),
        pl.col("total_tds_allowed").mean().alias("avg_total_tds_allowed"),
    ]).row(0, named=True)

    print(f"\n---- {opponent_team} Defense: Allowed per game (last 10 games) ----")
    print(f"games: {last10.height}")
    print(f"pass_yards_allowed: {float(avg['avg_pass_yards_allowed']):.3f}")
    print(f"rush_yards_allowed: {float(avg['avg_rush_yards_allowed']):.3f}")
    print(f"total_yards_allowed: {float(avg['avg_total_yards_allowed']):.3f}")
    print(f"pass_tds_allowed: {float(avg['avg_pass_tds_allowed']):.3f}")
    print(f"rush_tds_allowed: {float(avg['avg_rush_tds_allowed']):.3f}")
    print(f"total_tds_allowed: {float(avg['avg_total_tds_allowed']):.3f}")

    # Optional: show the actual last 10 game totals (readable, not too wide)
    last10_show = last10.select([
        "game_date", "season", "week", "season_type", "team",
        "pass_yards_allowed", "rush_yards_allowed", "total_yards_allowed",
        "pass_tds_allowed", "rush_tds_allowed", "total_tds_allowed",
    ])
    print_last_games_json(f"---- {opponent_team} Defense: Last 10 game totals ----", last10_show)


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
            print("\nClosest matches:")
            for name in suggestions["player_display_name"].to_list():
                print(" -", name)
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

    # Player vs opponent: all-time averages + last 5 games
    print_vs_opponent_summary_and_last5(player_games, opponent, stat_category)

    # Opponent defense: last 10 games allowed per game (approx, from your dataset)
    print_opponent_allowed_last10(df, opponent)


if __name__ == "__main__":
    main()
