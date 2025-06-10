import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_preprocess_data(
    file_path, train_test_split=0.8, standardization="iqr", debug=0
):
    """Load and preprocess energy consumption data from a CSV file.

    Args:
        file_path (str or Path): Path to the CSV data file
        train_test_split (float, optional): Fraction of data to use for training. Defaults to 0.8.
        standardization (str, optional): Standardization method to use, either "mean_std" or "iqr".
            Defaults to "iqr".
        debug (int, optional): If > 0, limits data to specified number of columns and rows for debugging.
            Defaults to 0.

    Returns:
        tuple: Contains:
            - df_train (pd.DataFrame): Standardized training data
            - df_test (pd.DataFrame): Standardized test data
            - unnormalize_fn (callable): Function to convert standardized values back to original scale

    Raises:
        AssertionError: If standardization method is invalid or train_test_split is not between 0 and 1
    """
    assert standardization in ["mean_std", "iqr"], "Invalid standardization method"
    assert train_test_split > 0 and train_test_split < 1, "Invalid train_test_split"

    use_cols = lambda x: x != "Unnamed: 0"
    if debug:
        use_cols = [x for x in range(1, 1 + debug)]

    df = pd.read_csv(
        file_path,
        delimiter=";",
        low_memory=False,
        decimal=",",
        usecols=use_cols,
    )

    df = df.astype(np.float32)
    if debug:
        offset = max(0, len(df) - debug * 1000)
        df = df.iloc[offset:]

    train_split_idx = int(len(df) * train_test_split)
    df_train = df.iloc[:train_split_idx]
    df_test = df.iloc[train_split_idx:]

    # Apply standardization
    if standardization == "mean_std":
        mu = df_train.values.mean()
        sigma = df_train.values.std()
        df_train = (df_train - mu) / sigma
        df_test = (df_test - mu) / sigma
        unnormalize_fn = lambda x: (
            (x * sigma + mu).astype(np.float32)
            if isinstance(x, (float, int))
            else (np.array(x) * sigma + mu).astype(np.float32)
        )
    elif standardization == "iqr":
        values = pd.Series(df_train.values.flatten())
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        df_train = (df_train - median) / iqr
        df_test = (df_test - median) / iqr
        unnormalize_fn = lambda x: (
            (x * iqr + median).astype(np.float32)
            if isinstance(x, (float, int))
            else (np.array(x) * iqr + median).astype(np.float32)
        )

    return df_train, df_test, unnormalize_fn


def plot_predictions(predictions, ground_truth, max_clients=5):
    """Plot model predictions against ground truth values for multiple clients.

    Args:
        predictions (dict): Dictionary mapping client IDs to their predicted values.
            Each value is a list/array of predictions.
        ground_truth (dict): Dictionary mapping client IDs to their actual values.
            Each value is a list/array of ground truth values.
        max_clients (int, optional): Maximum number of clients to plot. Defaults to 5.

    The function creates a single plot showing both predictions and ground truth
    for up to max_clients number of clients. For each client, the ground truth is
    shown as a solid line and predictions as a dashed line in the same color.
    """
    plt.figure(figsize=(12, 6))

    # Get list of client IDs and limit to max_clients
    client_ids = list(predictions.keys())[:max_clients]

    # Plot each client's predictions and ground truth
    for i, client_id in enumerate(client_ids):
        color = f"C{i}"  # Use same color for each client
        plt.plot(
            ground_truth[client_id], color=color, label=f"Client {client_id}", alpha=0.7
        )
        plt.plot(predictions[client_id], color=color, linestyle="--", alpha=0.7)

    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
