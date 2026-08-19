"""
models_lstm.py

Branch B from Section 2.2 of the methodology: an LSTM-FCN style sequence
model (Karim et al., 2018) trained on windowed meal/keeper-event
sequences built by features.build_sequences().

NOTE: requires `tensorflow` (not installed in the assignment sandbox
used to draft this repo — install locally with `pip install tensorflow`
before running this file). This does not affect the tabular pipeline
(models_tabular.py, baseline_model.py), which run on scikit-learn only.
"""

def build_lstm_fcn(n_timesteps: int, n_features: int):
    from tensorflow.keras import layers, models

    # Recurrent branch
    inp = layers.Input(shape=(n_timesteps, n_features))
    lstm_out = layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)(inp)

    # Convolutional branch (1-D FCN over the same sequence)
    conv = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Conv1D(128, 3, padding="same", activation="relu")(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.GlobalAveragePooling1D()(conv)

    merged = layers.Concatenate()([lstm_out, conv])
    merged = layers.Dropout(0.3)(merged)
    out = layers.Dense(1, activation="sigmoid")(merged)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", "AUC"])
    return model


if __name__ == "__main__":
    # Smoke test with dummy shapes matching features.build_sequences() output
    # (window=6, 4 sequence features).
    m = build_lstm_fcn(n_timesteps=6, n_features=4)
    m.summary()
