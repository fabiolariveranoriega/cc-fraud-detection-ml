from torch_geometric.data import Dataset
import torch

class GraphDataset(Dataset):
    def __init__(self, data, num_vertex_features:int = 16):
        super().__init__()
        self.data = data

        self.num_vertex_features = num_vertex_features
        #self.x = torch.zeros((torch.max(self.data.edge_index) + 1, num_vertex_features))

        self.edge_index = self.data.edge_index
        self.edge_attr = self.data.edge_attr
        self.targets = self.data.y.type(torch.float32)

    def len(self):
        return len(self.targets)

    def get(self, idx):
        return self.edge_index[:, idx], self.edge_attr[idx], self.targets[idx]
    
    def add_synthetic_data(self, synthetic_edge_index, synthetic_edge_attr, synthetic_targets):
        self.edge_index = torch.Tensor(synthetic_edge_index)
        self.edge_attr = torch.Tensor(synthetic_edge_attr)
        self.targets = torch.Tensor(synthetic_targets).type(torch.float32)