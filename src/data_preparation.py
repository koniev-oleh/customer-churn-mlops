import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Завантажує CSV-файл у DataFrame."""
    return pd.read_csv(path)

def prepare_features(df: pd.DataFrame, target_column: str = "Churn"):
    df = df.copy()
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Надійне перетворення Yes/No у 1/0
    y = y.replace({"No": 0, "Yes": 1})

    # One-hot encoding для ознак
    X = pd.get_dummies(X, drop_first=True)

    # Заповнення пропусків
    X = X.fillna(0)

    return X, y.astype(int)


if __name__ == "__main__":
    # Приклад використання
    df = load_data("data/raw/churn.csv")
    X, y = prepare_features(df, target_column="Churn")

    print("Ознаки (X):")
    print(X.head())
    print("\nЦільова змінна (y):")
    print(y.head())
