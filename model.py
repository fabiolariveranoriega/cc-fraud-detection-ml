import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, NNConv

class EdgeClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, activation_fn):
        super(EdgeClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim // 2))
            hidden_dim = hidden_dim // 2
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = activation_fn

        self.output_activation_fn = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.activation_fn(x)
            x = self.dropout(x)

        u, v = edge_index
        edge_features = torch.cat([x[u], x[v]], dim=-1)
        edge_features = self.dropout(edge_features)
       
        return self.output_activation_fn(self.fc(edge_features))

class EdgeClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, activation_fn, num_edges:int=10):
        super(EdgeClassifier, self).__init__()

        edge_nn = nn.Sequential(
            nn.Linear(num_edges, hidden_dim * hidden_dim), 
            activation_fn,
            nn.Linear(hidden_dim * hidden_dim, input_dim * hidden_dim)  
        )
        
        # Define NNConv layers
        self.layers = nn.ModuleList()
        self.layers.append(NNConv(input_dim, hidden_dim, nn=edge_nn, aggr='mean'))
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(10, hidden_dim * hidden_dim), 
                activation_fn,
                nn.Linear(hidden_dim * hidden_dim, hidden_dim * (hidden_dim // 2)) 
            )
            self.layers.append(NNConv(hidden_dim, hidden_dim // 2, nn=edge_nn, aggr='mean'))
            hidden_dim = hidden_dim // 2

        # Final fully connected layer for edge classification
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = activation_fn

        self.output_activation_fn = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):    
        # Pass through NNConv layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            #print(f"x after NNConv: {x.shape}")
            x = self.activation_fn(x)
            x = self.dropout(x)

        u, v = edge_index  # Get source and destination nodes of edges
        edge_features = torch.cat([x[u], x[v]], dim=-1)
        
        edge_features = self.dropout(edge_features)

        return self.output_activation_fn(self.fc(edge_features))