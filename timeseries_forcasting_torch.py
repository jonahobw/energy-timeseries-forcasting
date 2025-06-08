import numpy as np
from data_processing import load_and_preprocess_data, plot_predictions
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pformat
from collections import defaultdict

class EnergyDataset(Dataset):
    def __init__(
        self, df, unnormalize_fn, window_size=10, predict_ahead=1, step_size=1
    ):
        self.df = df
        self.unnormalize_fn = unnormalize_fn
        self.window_size = window_size
        self.predict_ahead = predict_ahead
        self.num_clients = df.shape[1]
        self.length = (len(df) - window_size - predict_ahead + 1) * self.num_clients
        self.step_size = step_size
        # Calculate how many samples we can get per client given the step_size
        max_start_idx = len(df) - window_size - predict_ahead
        self.samples_per_client = max(0, (max_start_idx // step_size) + 1)
        self.length = self.samples_per_client * self.num_clients

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Calculate which client and which sample within that client
        client_idx = idx // self.samples_per_client
        sample_idx = idx % self.samples_per_client

        # Calculate the actual time index using step_size
        time_idx = sample_idx * self.step_size
        data = self.df.iloc[time_idx : time_idx + self.window_size, client_idx].values
        target = self.df.iloc[
            time_idx
            + self.window_size : time_idx
            + self.window_size
            + self.predict_ahead,
            client_idx,
        ].values
        return data.reshape(-1, 1), target, client_idx

    def __repr__(self):
        instance_vars = {
            "window_size": self.window_size,
            "predict_ahead": self.predict_ahead,
            "num_clients": self.num_clients,
            "length": self.length,
            "step_size": self.step_size,
            "df_shape": self.df.shape,
            "samples_per_client": self.samples_per_client,
        }
        return pformat(instance_vars)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x): # x: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x) # lstm_out: (batch_size, seq_len, hidden_size)
        out = self.fc(lstm_out[:, -1, :])
        return out


def get_data_loader(
    df,
    unnormalize_fn,
    window_size=10,
    predict_ahead=1,
    batch_size=32,
    shuffle=False,
    show_stats=True,
    step_size=1,
):
    dataset = EnergyDataset(df, unnormalize_fn, window_size, predict_ahead, step_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=100,
    lr=0.001,
    weight_decay=0.0001,
    device="cpu",
):
    train_losses = []
    test_losses = []
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_test_loss = 0
        model.train()
        for data, target, _ in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            for data, target, _ in tqdm(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                epoch_test_loss += criterion(output, target).item()

        epoch_test_loss /= len(test_loader)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}"
        )
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)

    return model, train_losses, test_losses


def generate_predictions(model, test_loader, unnormalize_fn, device):
    model.eval()
    predictions = defaultdict(list)
    ground_truth = defaultdict(list)
    
    with torch.no_grad():
        for data, target, client_idx in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_batch = unnormalize_fn(output.cpu().numpy()) 
            truth_batch = unnormalize_fn(target.cpu().numpy())
            
            # Store predictions and ground truth for each client
            for i, client_id in enumerate(client_idx):
                client_id = client_id.item()
                predictions[client_id].append(pred_batch[i])
                ground_truth[client_id].append(truth_batch[i])
    
    return predictions, ground_truth


if __name__ == "__main__":
    WINDOW_SIZE = 64  # Number of time steps to use as input sequence
    PREDICT_AHEAD = 1  # Number of time steps to predict into the future
    STEP_SIZE = WINDOW_SIZE  # Number of time steps to skip between samples
    BATCH_SIZE = 32  # Number of samples per training batch
    DEVICE = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Use GPU if available, otherwise CPU
    HIDDEN_SIZE = 64  # Number of hidden units in LSTM layer
    DEBUG = 2  # 0 for no debug, otherwise use first N columns
    EPOCHS = 50  # Number of training epochs
    LR = 0.001  # Learning rate for optimizer
    WEIGHT_DECAY = 0.0001  # L2 regularization parameter

    print("Loading data...")
    datafile = Path(__file__).parent / "data" / "LD2011_2014.txt"
    df_train, df_test, unnormalize_fn = load_and_preprocess_data(datafile, debug=DEBUG)
    train_dl, train_dataset = get_data_loader(
        df_train,
        unnormalize_fn,
        window_size=WINDOW_SIZE,
        predict_ahead=PREDICT_AHEAD,
        batch_size=BATCH_SIZE,
        step_size=STEP_SIZE,
    )
    test_dl, test_dataset = get_data_loader(
        df_test,
        unnormalize_fn,
        window_size=WINDOW_SIZE,
        predict_ahead=PREDICT_AHEAD,
        batch_size=BATCH_SIZE,
    )
    print("\n\nDATASET STATS")
    print(train_dataset)
    print(test_dataset)
    print("\n\n")

    print("Creating model...")
    model = LSTM(input_size=1, hidden_size=HIDDEN_SIZE, output_size=PREDICT_AHEAD)

    print(f"Training model for {EPOCHS} epochs...")
    model, train_losses, test_losses = train_model(
        model,
        train_dl,
        test_dl,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
    )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.title("MSE Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    print("Generating predictions...")
    predictions, ground_truth = generate_predictions(model, test_dl, unnormalize_fn, DEVICE)
    plot_predictions(predictions, ground_truth)
