"""Load and run THOR's pretrained 50-model MLP ensemble, and compute the
per-point prediction entropy (confidence map) that goes with its regime
classification.
"""
import os
from pickle import load as pkl_load

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input


def build_model():
    inputs = Input(shape=(8,))
    h = Dense(24, activation="tanh")(inputs)
    h = Dense(24, activation="tanh")(h)
    h = Dense(16, activation="tanh")(h)
    h = Dense(16, activation="tanh")(h)
    outputs = Dense(6, activation="softmax")(h)
    return Model(inputs, outputs)


def load_scaler(model_dir):
    with open(os.path.join(model_dir, "scaler_v1.pkl"), "rb") as f:
        return pkl_load(f)


def scale(X_raw, scaler):
    return scaler.transform(X_raw).astype("float32")


def run(X, valid_mask, model_dir, n_models=50, batch_size=8192):
    """X is the already-scaled (n_valid, 8) feature matrix. Returns
    (regime_map, entropy_map), both (ny, nx) with NaN over land/invalid
    points. entropy_map is Shannon entropy of the ensemble-mean class
    probabilities, normalized by log2(n_classes) to [0, 1]: 0 = the
    ensemble is fully certain, 1 = maximally uncertain (uniform across
    classes).
    """
    model = build_model()
    preds = np.zeros((n_models, X.shape[0], 6), dtype="float32")
    for i in range(n_models):
        model.load_weights(os.path.join(model_dir, f"vanillamodel_{i}.h5"))
        preds[i] = model.predict(X, verbose=0, batch_size=batch_size)
        print(f"  ensemble member {i + 1}/{n_models} done")

    mean_probs = preds.mean(axis=0)

    regime_map = np.full(valid_mask.shape, np.nan)
    regime_map[valid_mask] = np.argmax(mean_probs, axis=1)

    n_classes = mean_probs.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(mean_probs > 0, mean_probs * np.log2(mean_probs), 0.0)
    entropy = -plogp.sum(axis=1) / np.log2(n_classes)
    entropy_map = np.full(valid_mask.shape, np.nan)
    entropy_map[valid_mask] = entropy

    return regime_map, entropy_map
