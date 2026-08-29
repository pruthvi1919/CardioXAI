"""
model.py
========
1D CNN for binary classification: Normal (0) vs PVC (1) beat.

Kept intentionally simple + shallow (3 conv blocks) so that:
  - it trains fast even on CPU (important with only 6 days left)
  - Grad-CAM on the LAST conv layer gives a clean, interpretable
    time-localized saliency map (this is what your XAI stage will use)
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_len=200, n_filters=(32, 64, 64)):
    inputs = layers.Input(shape=(input_len, 1), name="ecg_signal")

    x = layers.Conv1D(n_filters[0], 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(n_filters[1], 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # name this layer explicitly -- Grad-CAM will hook into it
    x = layers.Conv1D(n_filters[2], 3, padding="same", activation="relu",
                       name="last_conv")(x)
    x = layers.BatchNormalization()(x)

    gap = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(gap)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pvc_prob")(x)

    model = models.Model(inputs, outputs, name="pvc_1dcnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
