import pandas as pd

def load_and_preprocess_data(file_path, train_test_split=0.8, standardization="iqr"):
    assert standardization in ["mean_std", "iqr"], "Invalid standardization method"
    assert train_test_split > 0 and train_test_split < 1, "Invalid train_test_split"
    
    df = pd.read_csv(
        file_path,
        delimiter=";",
        low_memory=False,
        decimal=",",
        usecols=lambda x: x != "Unnamed: 0",
    )

    train_split_idx = int(len(df.columns) * train_test_split)
    df_train = df.iloc[:, :train_split_idx]
    df_test = df.iloc[:, train_split_idx:]

    # Apply standardization
    if standardization == "mean_std":
        mu = df_train.values.mean()
        sigma = df_train.values.std()
        df_train = (df_train - mu) / sigma
        df_test = (df_test - mu) / sigma
        unnormalize_fn = lambda x: x * sigma + mu
    elif standardization == "iqr":
        values = pd.Series(df_train.values.flatten())
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        df_train = (df_train - median) / iqr
        df_test = (df_test - median) / iqr
        unnormalize_fn = lambda x: x * iqr + median

    return df_train, df_test, unnormalize_fn