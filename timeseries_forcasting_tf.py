from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm

from data_processing import load_and_preprocess_data, plot_predictions


class EnergyDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        unnormalize_fn,
        window_size=10,
        predict_ahead=1,
        step_size=1,
        batch_size=32,
    ):
        super().__init__()  # Properly initialize parent class
        self.df = df
        self.unnormalize_fn = unnormalize_fn
        self.window_size = window_size
        self.predict_ahead = predict_ahead
        self.step_size = step_size
        self.batch_size = batch_size
        self.num_clients = df.shape[1]

        # Calculate total number of samples
        max_start_idx = len(df) - window_size - predict_ahead
        self.samples_per_client = max(0, (max_start_idx // step_size) + 1)
        self.total_samples = self.samples_per_client * self.num_clients

        # Create all possible indices
        self.indices = [
            (client_idx, sample_idx)
            for client_idx in range(self.num_clients)
            for sample_idx in range(self.samples_per_client)
        ]

    def __len__(self):
        # Return number of batches
        return (self.total_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_samples)
        batch_indices = self.indices[start_idx:end_idx]

        X_batch, y_batch = [], []

        for client_idx, sample_idx in batch_indices:
            time_idx = sample_idx * self.step_size
            data = self.df.iloc[
                time_idx : time_idx + self.window_size, client_idx
            ].values
            target = self.df.iloc[
                time_idx
                + self.window_size : time_idx
                + self.window_size
                + self.predict_ahead,
                client_idx,
            ].values
            X_batch.append(data.reshape(-1, 1))
            y_batch.append(target)

        return tf.convert_to_tensor(X_batch, dtype=tf.float32), tf.convert_to_tensor(
            y_batch, dtype=tf.float32
        )

    def get_client_data(self, client_idx):
        """Yields (X, y) batches for a single client."""
        assert 0 <= client_idx < self.num_clients, f"Invalid client index: {client_idx}"

        # Get all sample indices for this client
        client_indices = [
            (c_idx, s_idx) for c_idx, s_idx in self.indices if c_idx == client_idx
        ]

        for i in range(0, len(client_indices), self.batch_size):
            batch_indices = client_indices[i : i + self.batch_size]
            X_batch, y_batch = [], []

            for _, sample_idx in batch_indices:
                time_idx = sample_idx * self.step_size
                data = self.df.iloc[
                    time_idx : time_idx + self.window_size, client_idx
                ].values
                target = self.df.iloc[
                    time_idx
                    + self.window_size : time_idx
                    + self.window_size
                    + self.predict_ahead,
                    client_idx,
                ].values
                X_batch.append(data.reshape(-1, 1))
                y_batch.append(target)

            yield (
                tf.convert_to_tensor(X_batch, dtype=tf.float32),
                tf.convert_to_tensor(y_batch, dtype=tf.float32),
            )


def build_lstm_model(input_size, hidden_size, output_size):
    model = models.Sequential(
        [
            layers.Input(shape=(None, input_size)),
            layers.LSTM(hidden_size),
            layers.Dense(output_size),
        ]
    )
    return model


def generate_predictions(model, test_gen, unnormalize_fn):
    predictions = defaultdict(list)
    ground_truth = defaultdict(list)

    for client in tqdm(range(test_gen.num_clients)):

        for x, y in test_gen.get_client_data(client):
            pred_batch = model.predict(x)
            pred_batch = (
                pred_batch.numpy() if hasattr(pred_batch, "numpy") else pred_batch
            )
            y_batch = y.numpy() if hasattr(y, "numpy") else y
            predictions[client].extend(unnormalize_fn(pred_batch.flatten()))
            ground_truth[client].extend(unnormalize_fn(y_batch.flatten()))

    return predictions, ground_truth


if __name__ == "__main__":
    WINDOW_SIZE = 64
    PREDICT_AHEAD = 1
    STEP_SIZE = WINDOW_SIZE
    BATCH_SIZE = 32
    HIDDEN_SIZE = 64
    DEBUG = 2
    EPOCHS = 30
    LR = 0.001

    print("Loading data...")
    datafile = Path(__file__).parent / "data" / "LD2011_2014.txt"
    df_train, df_test, unnormalize_fn = load_and_preprocess_data(datafile, debug=DEBUG)

    print("Preparing datasets...")
    train_gen = EnergyDataset(
        df_train,
        unnormalize_fn,
        window_size=WINDOW_SIZE,
        predict_ahead=PREDICT_AHEAD,
        step_size=STEP_SIZE,
        batch_size=BATCH_SIZE,
    )
    test_gen = EnergyDataset(
        df_test,
        unnormalize_fn,
        window_size=WINDOW_SIZE,
        predict_ahead=PREDICT_AHEAD,
        batch_size=BATCH_SIZE,
    )

    print("Building model...")
    model = build_lstm_model(
        input_size=1, hidden_size=HIDDEN_SIZE, output_size=PREDICT_AHEAD
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mse")

    print("Training model...")
    history = model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Test Loss")
    plt.legend()
    plt.title("MSE Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    print("Generating predictions...")
    predictions, ground_truth = generate_predictions(model, test_gen, unnormalize_fn)
    plot_predictions(predictions, ground_truth)
