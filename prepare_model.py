# prep_all_stats_decay_only_final_train_through_2025.py
#
# Builds a unified dataset for ALL stat categories (stacked), with features from ALL prior games
# using exponential decay (no max-60 cap).
#
# Splits:
# - Dev Train: seasons <= 2023
# - Dev Val:   season == 2024  (use to tune tau/model size/etc.)
# - Final Train (for production): seasons <= 2025  (includes 2025 for best 2026 accuracy)
#
# No model training here; just builds arrays + builds the model structure.

import json
import numpy as np
import polars as pl
import tensorflow as tf


# ----------------------------
# Config
# ----------------------------
JSON_PATH = "nfl_player_gamelogs.json"

CATEGORIES = [
    # QB
    "passing_yards", "completions", "attempts", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    # RB/WR/TE
    "receiving_yards", "receptions", "receiving_tds",
]

TAU = 18.0
MIN_HISTORY_GAMES = 1

DEV_VAL_SEASON = 2024
DEV_TRAIN_MAX_SEASON = 2023
FINAL_TRAIN_MAX_SEASON = 2025


# ----------------------------
# Load JSON dataset
# ----------------------------
def load_json_dataset(path: str) -> pl.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pl.DataFrame(data)

    required = ["player_id", "opponent_team", "game_date", "season"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = (
        df.with_columns(
            pl.col("game_date")
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .alias("game_date")
        )
        .sort(["player_id", "game_date"])
    )

    for c in CATEGORIES:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).alias(c))
        df = df.with_columns(pl.col(c).fill_null(0).alias(c))

    return df


# ----------------------------
# Exp-decay features using ALL history
# ----------------------------
def exp_weights_all(n: int, tau: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float32)
    w = np.exp(-(n - 1 - idx) / tau)
    w = w / (w.sum() + 1e-8)
    return w


def build_features_for_series_all_history(values: np.ndarray, tau: float):
    T = len(values)
    values = values.astype(np.float32)

    ewm_mean = np.full(T, np.nan, dtype=np.float32)
    ewm_std  = np.full(T, np.nan, dtype=np.float32)
    last1    = np.full(T, np.nan, dtype=np.float32)
    mean5    = np.full(T, np.nan, dtype=np.float32)
    hist_len = np.full(T, np.nan, dtype=np.float32)

    for t in range(T):
        if t == 0:
            continue
        hist = values[:t]
        hist = np.nan_to_num(hist, nan=0.0)

        w = exp_weights_all(len(hist), tau)
        mu = float((hist * w).sum())
        var = float(((hist - mu) ** 2 * w).sum())
        sd = float(np.sqrt(var))

        ewm_mean[t] = mu
        ewm_std[t]  = sd
        last1[t]    = float(values[t - 1])

        last5 = hist[-5:] if len(hist) >= 5 else hist
        mean5[t] = float(last5.mean()) if len(last5) else 0.0

        hist_len[t] = float(len(hist))

    return ewm_mean, ewm_std, last1, mean5, hist_len


# ----------------------------
# Stack rows: (player, game, stat_category)
# ----------------------------
def build_stacked_training_arrays(df: pl.DataFrame, categories, tau: float, min_hist_games: int):
    pid_arr    = df["player_id"].to_numpy()
    opp_arr    = np.array(df["opponent_team"].to_list(), dtype=object)
    date_arr   = np.array(df["game_date"].to_list())
    season_arr = df["season"].to_numpy()

    boundaries = []
    start = 0
    for i in range(1, len(pid_arr) + 1):
        if i == len(pid_arr) or pid_arr[i] != pid_arr[start]:
            boundaries.append((start, i))
            start = i

    out_player, out_opp, out_cat, out_dates, out_seasons, out_X, out_y = [], [], [], [], [], [], []

    for (s, e) in boundaries:
        pid_slice   = pid_arr[s:e]
        opp_slice   = opp_arr[s:e]
        date_slice  = date_arr[s:e]
        seas_slice  = season_arr[s:e]

        for cat in categories:
            vals = df[cat].slice(s, e - s).to_numpy().astype(np.float32)

            ewm_mean, ewm_std, last1, mean5, hist_len = build_features_for_series_all_history(vals, tau)

            mask = ~np.isnan(ewm_mean) & (hist_len >= float(min_hist_games))
            if mask.sum() == 0:
                continue

            X = np.stack([ewm_mean[mask], ewm_std[mask], last1[mask], mean5[mask], hist_len[mask]], axis=1).astype(np.float32)
            y = vals[mask].astype(np.float32)

            out_player.append(pid_slice[mask])
            out_opp.append(opp_slice[mask])
            out_cat.append(np.array([cat] * int(mask.sum()), dtype=object))
            out_dates.append(date_slice[mask])
            out_seasons.append(seas_slice[mask])
            out_X.append(X)
            out_y.append(y)

    player_ids = np.concatenate(out_player)
    opp_teams  = np.concatenate(out_opp)
    stat_cats  = np.concatenate(out_cat)
    game_dates = np.concatenate(out_dates)
    seasons    = np.concatenate(out_seasons)
    X_num      = np.concatenate(out_X)
    y_all      = np.concatenate(out_y)

    return player_ids, opp_teams, stat_cats, seasons, game_dates, X_num, y_all


# ----------------------------
# Dev split + Final train split
# ----------------------------
def dev_and_final_splits(player_ids, opp_teams, stat_cats, seasons, X_num, y):
    # Dev: train on <= 2023, validate on 2024
    dev_train_idx = seasons <= DEV_TRAIN_MAX_SEASON
    dev_val_idx   = seasons == DEV_VAL_SEASON

    # Final: train on <= 2025 (this is what you use to predict 2026)
    final_train_idx = seasons <= FINAL_TRAIN_MAX_SEASON

    def sl(mask):
        return (
            player_ids[mask],
            opp_teams[mask],
            stat_cats[mask],
            X_num[mask],
            y[mask],
        )

    return sl(dev_train_idx), sl(dev_val_idx), sl(final_train_idx)


# ----------------------------
# Build ONE TF model
# ----------------------------
def build_model(p_vocab_source: np.ndarray, o_vocab_source: np.ndarray, c_vocab_source: np.ndarray, num_feat_dim: int) -> tf.keras.Model:
    player_vocab = np.unique(p_vocab_source)
    team_vocab   = np.unique(o_vocab_source)
    cat_vocab    = np.unique(c_vocab_source)

    player_lookup = tf.keras.layers.StringLookup(vocabulary=player_vocab, mask_token=None, num_oov_indices=1)
    team_lookup   = tf.keras.layers.StringLookup(vocabulary=team_vocab,   mask_token=None, num_oov_indices=1)
    cat_lookup    = tf.keras.layers.StringLookup(vocabulary=cat_vocab,    mask_token=None, num_oov_indices=1)

    inp_player = tf.keras.Input(shape=(), dtype=tf.string, name="player_id")
    inp_opp    = tf.keras.Input(shape=(), dtype=tf.string, name="opponent_team")
    inp_cat    = tf.keras.Input(shape=(), dtype=tf.string, name="stat_category")
    inp_num    = tf.keras.Input(shape=(num_feat_dim,), dtype=tf.float32, name="num_feats")

    p_id = player_lookup(inp_player)
    t_id = team_lookup(inp_opp)
    c_id = cat_lookup(inp_cat)

    p_emb = tf.keras.layers.Embedding(player_lookup.vocabulary_size(), 16)(p_id)
    t_emb = tf.keras.layers.Embedding(team_lookup.vocabulary_size(),   8)(t_id)
    c_emb = tf.keras.layers.Embedding(cat_lookup.vocabulary_size(),    4)(c_id)

    x = tf.keras.layers.Concatenate()([inp_num, p_emb, t_emb, c_emb])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1, name="pred")(x)

    model = tf.keras.Model(
        inputs={"player_id": inp_player, "opponent_team": inp_opp, "stat_category": inp_cat, "num_feats": inp_num},
        outputs=out,
    )

    model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

    return model


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    df = load_json_dataset(JSON_PATH)
    print("Rows:", df.height)

    print("Stacking categories + building exp-decay features...")
    player_ids, opp_teams, stat_cats, seasons, game_dates, X_num, y = build_stacked_training_arrays(
        df, CATEGORIES, TAU, MIN_HISTORY_GAMES
    )

    # Dev + Final splits
    (p_tr, o_tr, c_tr, X_tr, y_tr), (p_va, o_va, c_va, X_va, y_va), (p_final, o_final, c_final, X_final, y_final) = dev_and_final_splits(
        player_ids, opp_teams, stat_cats, seasons, X_num, y
    )

    print("Shapes:")
    print(f"  Dev Train (<= {DEV_TRAIN_MAX_SEASON}):", X_tr.shape, y_tr.shape)
    print(f"  Dev Val   (== {DEV_VAL_SEASON}):", X_va.shape, y_va.shape)
    print(f"  Final Train (<= {FINAL_TRAIN_MAX_SEASON}):", X_final.shape, y_final.shape)

    print("Building model (vocab from Dev Train)...")
    model = build_model(p_tr, o_tr, c_tr, num_feat_dim=X_tr.shape[1])
    model.summary()

    # Save arrays for later training
    np.savez(
        "arrays_dev_train_val_and_final_train.npz",
        p_tr=p_tr, o_tr=o_tr, c_tr=c_tr, X_tr=X_tr, y_tr=y_tr,
        p_va=p_va, o_va=o_va, c_va=c_va, X_va=X_va, y_va=y_va,
        p_final=p_final, o_final=o_final, c_final=c_final, X_final=X_final, y_final=y_final,
    )
    print("Saved arrays to arrays_dev_train_val_and_final_train.npz")
