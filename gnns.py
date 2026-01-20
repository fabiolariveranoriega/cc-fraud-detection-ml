import os
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-bb893506-cb4d-5e7c-a4ad-920e2a3471f5"



import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, NNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.nn import Linear
from torch.utils.data import Subset
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from functools import partial
from torch_geometric.loader import DataLoader
import pandas as pd 
from torch_geometric.utils import from_networkx
import networkx as nx
from tqdm import tqdm
from sklearn import preprocessing

from torch_geometric.data import Dataset, Data


import optuna
import joblib  # For saving the best model


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


path = 'credit_card_transactions-ibm_v2.csv'
data = pd.read_csv(path)


N = 25
fraud_df = data[data['Is Fraud?'] == 'Yes']
non_fraud_df = data[data['Is Fraud?'] == 'No'].groupby('User').apply(lambda x: x.head(N)).reset_index(drop=True)
data = pd.concat([fraud_df, non_fraud_df]).reset_index(drop=True)



merchants_with_multiple_states = data.groupby('Merchant Name')['Merchant City'].nunique()
merchants_with_multiple_states = merchants_with_multiple_states[merchants_with_multiple_states > 1]
print(merchants_with_multiple_states)
data['Merchant Name & City'] = data['Merchant Name'].astype(str)+ '_' + data['Merchant City']
data['Merchant Name & City'].head(1)
data['Amount'] = data['Amount'].replace('[\$,]', '', regex=True).astype(float)
data[['hour', 'minute']] = data['Time'].str.split(':', expand=True).astype(int)
data.drop(['Merchant State', 'Zip'], axis = 1, inplace = True)
data['Errors?'].fillna(value = 'None', inplace = True)


data.drop(['Time', 'Merchant Name', 'Merchant City'], axis = 1, inplace = True)


# encoding 
label_encoder = preprocessing.LabelEncoder()
data['Use Chip'] = label_encoder.fit_transform(data['Use Chip'])
print(data['Use Chip'])


label_encoder = preprocessing.LabelEncoder()
data['Errors?'] = label_encoder.fit_transform(data['Errors?'])
data['Errors?'].unique() 



label_encoder = preprocessing.LabelEncoder()
data['Is Fraud?'] = label_encoder.fit_transform(data['Is Fraud?'])
data['Is Fraud?'].head(5)


data.info()


user_ids = data['User'].unique()
merchant_ids = data['Merchant Name & City'].unique()

# Create dictionaries to map users and merchants to unique indices
user_mapping = {user: idx for idx, user in enumerate(user_ids)}
merchant_mapping = {merchant: idx + len(user_ids) for idx, merchant in enumerate(merchant_ids)}

# Step 2: Initialize NetworkX MultiGraph for undirected edges with multiple instances allowed between nodes
G = nx.MultiGraph()

# Step 3: Create nodes and edges with attributes
for _, row in tqdm(data.iterrows(), total=len(data)):
    # Map user and merchant to their unique indices
    user_idx = user_mapping[row['User']]
    merchant_idx = merchant_mapping[row['Merchant Name & City']]
    
    # Define edge attributes
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
    
    # Add the edge from user to merchant with attributes; multiple edges between nodes are allowed
    G.add_edge(user_idx, merchant_idx, **edge_attributes)

# Step 4: Convert NetworkX MultiGraph to PyTorch Geometric Data
# Extract edge indices (source and target nodes) and transpose to match PyTorch Geometric format
edge_index = torch.tensor([(u, v) for u, v, _ in G.edges], dtype=torch.long).t().contiguous()

# Extract edge attributes into tensors
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

# Stack edge attributes into a single tensor
edge_attr = torch.stack(list(edge_attrs.values()), dim=1)
edge_labels = torch.tensor([float(G[u][v][k]["is_fraud"]) for u, v, k in G.edges(keys=True)], dtype=torch.long)

# Create the PyTorch Geometric Data object
data = Data(edge_index=edge_index, edge_attr=edge_attr, y = edge_labels)


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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, activation_fn):
        super(EdgeClassifier, self).__init__()

        # Define the neural network for edge features (edge_nn)
        # edge_attr has 10 features, so input to the first layer should be 10
        edge_nn = nn.Sequential(
            nn.Linear(10, hidden_dim * hidden_dim),  # Now the input dimension is 10 (the number of edge features)
            activation_fn,
            nn.Linear(hidden_dim * hidden_dim, input_dim * hidden_dim)  # Output the transformation matrix for each edge
        )
        
        # Define NNConv layers
        self.layers = nn.ModuleList()
        self.layers.append(NNConv(input_dim, hidden_dim, nn=edge_nn, aggr='mean'))
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(10, hidden_dim * hidden_dim),  # Now the input dimension is 10 (the number of edge features)
                activation_fn,
                nn.Linear(hidden_dim * hidden_dim, hidden_dim * (hidden_dim // 2))  # Output the transformation matrix for each edge
            )
            self.layers.append(NNConv(hidden_dim, hidden_dim // 2, nn=edge_nn, aggr='mean'))
            hidden_dim = hidden_dim // 2

        # Final fully connected layer for edge classification
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = activation_fn

        self.output_activation_fn = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        # Print shapes for debugging
        #print(f"x.shape: {x.shape}")
        #print(f"edge_index.shape: {edge_index.shape}")
        #print(f"edge_attr.shape: {edge_attr.shape}")
        
        # Pass through NNConv layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            #print(f"x after NNConv: {x.shape}")
            x = self.activation_fn(x)
            x = self.dropout(x)

        u, v = edge_index  # Get source and destination nodes of edges
        edge_features = torch.cat([x[u], x[v]], dim=-1)
        #print(f"edge_features.shape: {edge_features.shape}")
        
        edge_features = self.dropout(edge_features)

        return self.output_activation_fn(self.fc(edge_features))



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

def objective(trial, dataset):
    # Hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    batch_size = trial.suggest_int('batch_size', 32, 128) 
    num_epochs = trial.suggest_int('num_epochs', 10, 50)
    num_layers = trial.suggest_int("num_layers", 2, 5)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.1)
    lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
    activation_fn = nn.Tanh() #trial.suggest_categorical("activation_fn", [F.relu, F.leaky_relu, F.elu, F.tanh]) #trial.suggest_categorical("activation_fn", [nn.Tanh()]) #trial.suggest_categorical("activation_fn", [F.relu, F.leaky_relu, F.elu, F.tanh])

    print(f"hidden_dim: {hidden_dim}, batch_size: {batch_size}, num_epochs: {num_epochs}, num_layers: {num_layers}, dropout_rate: {dropout_rate}, lr: {lr}, activation_fn: {activation_fn}")

        
    # Lists to store metrics for each fold
    all_fold_accuracies = []
    all_fold_precisions = []
    all_fold_recalls = []
    all_fold_f1s = []


    output_dim = 1
    k_folds = 5
    # Initialize KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)


    # k-fold cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset.y)))):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split dataset into training and validation sets for the current fold
        train_dataset = Subset(GraphDataset(dataset), train_idx)
        val_dataset = Subset(GraphDataset(dataset), val_idx)

        # Create DataLoaders for train and validation sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # SMOTE
        train_features = [train_dataset.dataset.data.edge_attr[i] for i in train_idx]
        train_labels = [train_dataset.dataset.data.y[i] for i in train_idx]

        scaler = StandardScaler()
        train_features = scaler.fit_transform(np.array(train_features))

        smote = SMOTE(sampling_strategy="auto", random_state=1337)
        synthetic_edge_attr, synthetic_labels = smote.fit_resample(train_features, train_labels)
        #print(f'train_labels: {train_labels}')
        #print(f'synthetic_labels: {synthetic_labels}')

        max_node_index = train_dataset.dataset.data.edge_index.max().item()
        num_edges = len(synthetic_edge_attr)

        u = torch.randint(0, max_node_index + 1, (num_edges,))
        v = torch.empty(num_edges, dtype=torch.long)

        for i in range(num_edges):
            v_i = torch.randint(0, max_node_index + 1, (1,))
            while v_i == u[i]: 
                v_i = torch.randint(0, max_node_index + 1, (1,))
            v[i] = v_i
        synthetic_edge_index = torch.stack([u, v], dim=0).contiguous()

        train_loader.dataset.dataset.add_synthetic_data(synthetic_edge_index, synthetic_edge_attr, synthetic_labels)


        # Initialize the model and optimizer (reset for each fold)
        input_dim = train_dataset.dataset.num_vertex_features
        model = EdgeClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        # Training loop for the current fold
        print("Train")
        for epoch in tqdm(range(num_epochs)):
            model.train()
            for edge_index, edge_attr, y in train_loader:
                edge_index, edge_attr, y = edge_index.to(device), edge_attr.to(device), y.to(device)
                #edge_index = edge_index.T
                unique_nodes = torch.unique(edge_index)  # Get unique nodes from the edge_index
                node_mapping = {old.item(): new for new, old in enumerate(unique_nodes)}  # Map old index to new sequential index
                
                # Reindex the edge_index based on the mapping
                edge_index = torch.tensor([[node_mapping[u.item()], node_mapping[v.item()]] for u, v in edge_index], dtype=torch.long).T
                edge_index = edge_index.to(device)


                x = torch.ones((len(unique_nodes), 16))
                x = x.to(device)
                optimizer.zero_grad()
                out = model(x, edge_index, edge_attr).reshape((-1,))
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                pred = (out > 0.5).type(torch.long)
                #print(f'y: {y}')

        # Evaluate the model on the validation set
        print("Val")
        model.eval() 
        val_preds, val_labels = [], []
        with torch.no_grad():
            for edge_index, edge_attr, y in val_loader:
                edge_attr = torch.Tensor(scaler.transform(edge_attr))
                edge_index, edge_attr, y = edge_index.to(device), edge_attr.to(device), y.to(device)
                #edge_index = edge_index.T
                unique_nodes = torch.unique(edge_index)  # Get unique nodes from the edge_index
                node_mapping = {old.item(): new for new, old in enumerate(unique_nodes)}  # Map old index to new sequential index
                
                # Reindex the edge_index based on the mapping
                edge_index = torch.tensor([[node_mapping[u.item()], node_mapping[v.item()]] for u, v in edge_index], dtype=torch.long).T
                edge_index = edge_index.to(device)


                x = torch.ones((len(unique_nodes), 16))
                x = x.to(device)
                out = model(x, edge_index, edge_attr).reshape((-1,))
                pred = (out > 0.5).type(torch.long)
                #print(f'y: {y}')
                val_preds.append(pred.cpu())
                val_labels.append(y.cpu())

        # Concatenate predictions and labels across batches
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        #print(f'val_labels:{val_labels}')

        # Calculate metrics for the current fold
        fold_accuracy = accuracy_score(val_labels, val_preds)
        fold_precision = precision_score(val_labels, val_preds)
        fold_recall = recall_score(val_labels, val_preds)
        fold_f1 = f1_score(val_labels, val_preds)

        print(f"Fold {fold + 1} - Accuracy: {fold_accuracy:.4f}, Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}, F1: {fold_f1:.4f}")
        joblib.dump(model, f"edge_classifier_fold={fold+1}_hidden_dim={hidden_dim}_batch_size_={batch_size}_num_epochs={num_epochs}_num_layers={num_layers}_dropout_rate={dropout_rate}_lr={lr}_activation_fn={activation_fn}.pkl")

        # Append metrics to lists
        all_fold_accuracies.append(fold_accuracy)
        all_fold_precisions.append(fold_precision)
        all_fold_recalls.append(fold_recall)
        all_fold_f1s.append(fold_f1)

    # Average metrics across all folds
    avg_accuracy = np.mean(all_fold_accuracies)
    avg_precision = np.mean(all_fold_precisions)
    avg_recall = np.mean(all_fold_recalls)
    avg_f1 = np.mean(all_fold_f1s)

    print("K-Fold Cross-Validation Results:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    return avg_f1


# new cellls 

# Define parameters

print('#'* 50)



# Lists to store metrics for each fold
all_fold_accuracies = []
all_fold_precisions = []
all_fold_recalls = []
all_fold_f1s = []

# Define the EdgeClassifier model with hyperparameters

# Running Optuna for hyperparameter optimization
input_dim = data.num_node_features
output_dim = 1
#train_mask, test_mask = # Define your train/test masks here as before

study = optuna.create_study(direction="maximize")


objective_with_args = partial(objective, dataset = data)

study.optimize(objective_with_args, n_trials=20)

# Best trial
best_trial = study.best_trial
print(f"Best trial's F1 score: {best_trial.value}")
print("Best hyperparameters:", best_trial.params)

# Train the final model with the best hyperparameters
best_params = best_trial.params
k_folds = 5
    # Initialize KFold
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)


for fold, (train_idx, val_idx) in enumerate(kf.split(data.y.numpy())):
    print(f"Fold {fold + 1}/{k_folds}")

    # Split dataset into training and validation sets for the current fold
    train_dataset = Subset(GraphDataset(data), train_idx)
    val_dataset = Subset(GraphDataset(data), val_idx)

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

    # SMOTE
    train_features = [train_dataset.dataset.data.edge_attr[i] for i in train_idx]
    train_labels = [train_dataset.dataset.data.y[i] for i in train_idx]

    scaler = StandardScaler()
    train_features = scaler.fit_transform(np.array(train_features))

    smote = SMOTE(sampling_strategy="auto", random_state=1337)
    synthetic_edge_attr, synthetic_labels = smote.fit_resample(train_features, train_labels)

    max_node_index = train_dataset.dataset.data.edge_index.max().item()
    num_edges = len(synthetic_edge_attr)

    u = torch.randint(0, max_node_index + 1, (num_edges,))
    v = torch.empty(num_edges, dtype=torch.long)

    for i in range(num_edges):
        v_i = torch.randint(0, max_node_index + 1, (1,))
        while v_i == u[i]: 
            v_i = torch.randint(0, max_node_index + 1, (1,))
        v[i] = v_i
    synthetic_edge_index = torch.stack([u, v], dim=0).contiguous()

    train_loader.dataset.dataset.add_synthetic_data(synthetic_edge_index, synthetic_edge_attr, synthetic_labels)

    # Initialize the model and optimizer (reset for each fold)
    model = EdgeClassifier(
        input_dim=input_dim,
        hidden_dim=best_params['hidden_dim'],
        output_dim=output_dim,
        num_layers=best_params['num_layers'],
        dropout_rate=best_params['dropout_rate'],
        activation_fn=best_params['activation_fn']
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = torch.nn.BCELoss()

    # Training loop for the current fold
    print("Train")
    for epoch in tqdm(range(best_params["num_epochs"])):
        model.train()
        for edge_index, edge_attr, y in train_loader:
            edge_index, edge_attr, y = edge_index.to(device), edge_attr.to(device), y.to(device)
            unique_nodes = torch.unique(edge_index)  # Get unique nodes from the edge_index
            node_mapping = {old.item(): new for new, old in enumerate(unique_nodes)}  # Map old index to new sequential index
            
            # Reindex the edge_index based on the mapping
            edge_index = torch.tensor([[node_mapping[u.item()], node_mapping[v.item()]] for u, v in edge_index], dtype=torch.long).T
            edge_index = edge_index.to(device)


            x = torch.ones((len(unique_nodes), 16))
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr).reshape((-1,))
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the validation set
    model.eval() 
    val_preds, val_labels = [], []
    print("Val")
    with torch.no_grad():
        for edge_index, edge_attr, y in val_loader:
            edge_attr = torch.Tensor(scaler.transform(edge_attr))
            edge_index, edge_attr, y = edge_index.to(device), edge_attr.to(device), y.to(device)
            unique_nodes = torch.unique(edge_index)  # Get unique nodes from the edge_index
            node_mapping = {old.item(): new for new, old in enumerate(unique_nodes)}  # Map old index to new sequential index
            
            # Reindex the edge_index based on the mapping
            edge_index = torch.tensor([[node_mapping[u.item()], node_mapping[v.item()]] for u, v in edge_index], dtype=torch.long).T
            edge_index = edge_index.to(device)


            x = torch.ones((len(unique_nodes), 16))
            x = x.to(device)
            out = model(x, edge_index, edge_attr).reshape((-1,))
            pred = (out > 0.5).type(torch.long)
            val_preds.append(pred.cpu())
            val_labels.append(y.cpu())

    # Concatenate predictions and labels across batches
    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)

    # Calculate metrics for the current fold
    fold_accuracy = accuracy_score(val_labels, val_preds)
    fold_precision = precision_score(val_labels, val_preds)
    fold_recall = recall_score(val_labels, val_preds)
    fold_f1 = f1_score(val_labels, val_preds)

    print(f"Fold {fold + 1} - Accuracy: {fold_accuracy:.4f}, Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}, F1: {fold_f1:.4f}")
    joblib.dump(model, f"best_edge_classifier_fold={fold+1}.pkl")

    # Append metrics to lists
    all_fold_accuracies.append(fold_accuracy)
    all_fold_precisions.append(fold_precision)
    all_fold_recalls.append(fold_recall)
    all_fold_f1s.append(fold_f1)

# Average metrics across all folds
avg_accuracy = np.mean(all_fold_accuracies)
avg_precision = np.mean(all_fold_precisions)
avg_recall = np.mean(all_fold_recalls)
avg_f1 = np.mean(all_fold_f1s)

print("K-Fold Cross-Validation Results:")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")

# Save the best model

