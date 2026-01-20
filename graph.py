from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx
from tqdm import tqdm

import torch
from torch_geometric.data import Data


class GNN(ABC):
    def __init__(self):
        pass

    def create_graph(self, data:pd.DataFrame, v0_mapping:dict, v1_mapping:dict):
        raise NotImplementedError("`create_graph` not implemented.")
    

class CreditCardGNN(GNN):
    def __init__(self):
        super().__init__()

    def create_graph(self, data, v0_mapping, v1_mapping):                
        G = nx.MultiGraph()

        for _, row in tqdm(data.iterrows(), total=len(data)):
            user_idx = v0_mapping[row['User']]
            merchant_idx = v1_mapping[row['Merchant Name & City']]
            
            edge_attributes = {
                "card": row["Card"],
                "amount": row["Amount"],
                "use_chip": row["Use Chip"],
                "errors": row["Errors?"],
                "is_fraud": row["Is Fraud?"],
                "hour": row["hour"],
                "minute": row["minute"],
                "year": row["Year"],
                "month": row["Month"],
                "day": row["Day"],
                "mcc": row["MCC"]
            }
            
            G.add_edge(user_idx, merchant_idx, **edge_attributes)

        return G

    def to_torch(self, G:nx.MultiGraph):
        edge_index = torch.tensor([(u, v) for u, v, _ in G.edges], dtype=torch.long).t().contiguous()

        edge_attrs = {
            "card": torch.tensor([float(G[u][v][k]["card"]) for u, v, k in G.edges(keys=True)]),
            "amount": torch.tensor([float(G[u][v][k]["amount"]) for u, v, k in G.edges(keys=True)]),
            "use_chip": torch.tensor([float(G[u][v][k]["use_chip"]) for u, v, k in G.edges(keys=True)]),
            "errors": torch.tensor([float(G[u][v][k]["errors"]) for u, v, k in G.edges(keys=True)]),
            
            "hour": torch.tensor([float(G[u][v][k]["hour"]) for u, v, k in G.edges(keys=True)]),
            "minute": torch.tensor([float(G[u][v][k]["minute"]) for u, v, k in G.edges(keys=True)]),
            "year": torch.tensor([float(G[u][v][k]["year"]) for u, v, k in G.edges(keys=True)]),
            "month": torch.tensor([float(G[u][v][k]["month"]) for u, v, k in G.edges(keys=True)]),
            "day": torch.tensor([float(G[u][v][k]["day"]) for u, v, k in G.edges(keys=True)]),
            "mcc": torch.tensor([float(G[u][v][k]["mcc"]) for u, v, k in G.edges(keys=True)]),
        }

        edge_attr = torch.stack(list(edge_attrs.values()), dim=1)
        edge_labels = torch.tensor([float(G[u][v][k]["is_fraud"]) for u, v, k in G.edges(keys=True)], dtype=torch.long)

        data = Data(edge_index=edge_index, edge_attr=edge_attr, y = edge_labels)

        return data