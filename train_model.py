# train_model_from_npz.py
#
# Trains the unified model on dev train (<=2023) and validates on 2024.
# Then (optional) retrains a final production model on data through 2025.
#
# Input: arrays_dev_train_val_and_final_train.npz
# Output:
#   - model_dev.keras
#   - model_final.keras (optional; used for 2026 predictions)

import numpy as np
import tensorflow as tf


NPZ_PATH = "arrays_dev_train_val_and_final_train.npz"

# Training hyperparams (tune these later)
BATCH_SIZE = 4096
EPOCHS_DEV = 25
EPOCHS_FINAL = 10  # final model often needs fewer epochs since you already tuned
LR = 1e-3

# If True, trains the final production model on <=2025
TRAIN_FINAL_MODEL = True


def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def make_tf_dataset(p, o, c, X, y, batch_size: int, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices((
        {
            "player_id": tf.convert_to_tensor(p, dtype=tf.string),
            "opponent_team": tf.convert_to_tensor(o, dtype=tf.string),
            "stat_category": tf.convert_to_tensor(c, dtype=tf.string),
            "num_feats": tf.convert_to_tensor(X, dtype=tf.float32),
        },
        tf.convert_to_tensor(y, dtype=tf.float32),
    ))
    if shuffle:
        ds = ds.shuffle(min(len(y), 200_000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(player_vocab, team_vocab, cat_vocab, num_feat_dim: int) -> tf.keras.Model:
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
        inputs={
            "player_id": inp_player,
            "opponent_team": inp_opp,
            "stat_category": inp_cat,
            "num_feats": inp_num,
        },
        outputs=out
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="mse",
        metrics=["mae"],   # works across TF/Keras versions
    )
    return model


def train_dev_and_save(data):
    p_tr, o_tr, c_tr, X_tr, y_tr = data["p_tr"], data["o_tr"], data["c_tr"], data["X_tr"], data["y_tr"]
    p_va, o_va, c_va, X_va, y_va = data["p_va"], data["o_va"], data["c_va"], data["X_va"], data["y_va"]

    # Build vocab from DEV TRAIN (so dev eval simulates rookies/unseen players)
    player_vocab = np.unique(p_tr)
    team_vocab   = np.unique(o_tr)
    cat_vocab    = np.unique(c_tr)

    model = build_model(player_vocab, team_vocab, cat_vocab, num_feat_dim=X_tr.shape[1])

    ds_tr = make_tf_dataset(p_tr, o_tr, c_tr, X_tr, y_tr, BATCH_SIZE, shuffle=True)
    ds_va = make_tf_dataset(p_va, o_va, c_va, X_va, y_va, BATCH_SIZE, shuffle=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=EPOCHS_DEV,
        callbacks=callbacks,
        verbose=1
    )

    model.save("model_dev.keras")
    print("Saved dev model to model_dev.keras")

    # Evaluate dev model on 2024 validation set
    val_metrics = model.evaluate(ds_va, verbose=1)
    print("Dev validation metrics (loss, mae):", val_metrics)

    return model, history


def train_final_and_save(data):
    p_final, o_final, c_final, X_final, y_final = (
        data["p_final"], data["o_final"], data["c_final"], data["X_final"], data["y_final"]
    )

    # For the production model, build vocab from FINAL TRAIN (includes 2024-2025 players/teams)
    player_vocab = np.unique(p_final)
    team_vocab   = np.unique(o_final)
    cat_vocab    = np.unique(c_final)

    model = build_model(player_vocab, team_vocab, cat_vocab, num_feat_dim=X_final.shape[1])

    ds_final = make_tf_dataset(p_final, o_final, c_final, X_final, y_final, BATCH_SIZE, shuffle=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=1, min_lr=1e-5),
    ]

    model.fit(ds_final, epochs=EPOCHS_FINAL, callbacks=callbacks, verbose=1)

    model.save("model_final.keras")
    print("Saved final (production) model to model_final.keras")
    return model


if __name__ == "__main__":
    data = load_npz(NPZ_PATH)

    # 1) Train dev model and validate on 2024
    train_dev_and_save(data)

    # 2) Train final model on data through 2025 (for 2026 predictions)
    if TRAIN_FINAL_MODEL:
        train_final_and_save(data)

    print("Done.")
