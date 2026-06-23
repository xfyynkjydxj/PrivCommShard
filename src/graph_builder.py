import pandas as pd
import networkx as nx


WINDOW_SIZE = 10000
INPUT_PATH = "../data/ethereum_clean.txt"


class TransactionGraphBuilder:

    def __init__(self):
        self.df = pd.read_csv(INPUT_PATH, sep="\t")

    def build_window_graph(self, start_idx):
        end_idx = start_idx + WINDOW_SIZE

        window_df = self.df.iloc[start_idx:end_idx]

        G = nx.Graph()

        for _, row in window_df.iterrows():
            sender = row["From"]
            receiver = row["To"]

            if G.has_edge(sender, receiver):
                G[sender][receiver]["weight"] += 1
            else:
                G.add_edge(sender, receiver, weight=1)

        return G

    def generate_all_windows(self):
        graphs = []

        total = len(self.df)

        for start in range(0, total, WINDOW_SIZE):
            G = self.build_window_graph(start)
            graphs.append(G)

            print(f"Window {len(graphs)} generated")

        return graphs


if __name__ == "__main__":
    builder = TransactionGraphBuilder()
    graphs = builder.generate_all_windows()

    print("Total windows:", len(graphs))
