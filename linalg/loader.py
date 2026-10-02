import pandas as pd

COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]


def load_fd001(path="data/train_FD001.txt"):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=125)
    return df

if __name__ == "__main__":
    df = load_fd001()
    print("Shape:", df.shape)
    print("Engines:", df["unit"].nunique())
    print("\nStd dev per sensor (near-0 = useless sensor):")
    print(df.filter(like="s").std().sort_values())
