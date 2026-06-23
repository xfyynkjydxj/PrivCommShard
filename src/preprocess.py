import pandas as pd


INPUT_PATH = "../data/ethereum_raw.csv"
OUTPUT_PATH = "../data/ethereum_clean.txt"


def preprocess_data():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_PATH)

    print("Original size:", len(df))

    df = df[df["isError"] == 0]

    df = df.dropna(subset=["From", "To"])

    df = df[df["From"] != df["To"]]

    df = df.sort_values(by="TimeStamp")

    df = df.reset_index(drop=True)

    print("Clean size:", len(df))

    df.to_csv(OUTPUT_PATH, index=False, sep="\t")

    print("Saved cleaned dataset.")


if __name__ == "__main__":
    preprocess_data()
