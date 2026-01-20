import torch
import optuna
import joblib  
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from .config import device, HIDDEN_DIM, BATCH_SIZE, NUM_EPOCHS, NUM_LAYERS, DROPOUT_RATE, LR, ACTIVATION_FN, NUM_FOLDS, SEED
from .dataset import GraphDataset
from .model import EdgeClassifier

def objective(trial, dataset):
    # Hyperparameters to tune
    hidden_dim = trial.suggest_int(*HIDDEN_DIM)
    batch_size = trial.suggest_int(*BATCH_SIZE) 
    num_epochs = trial.suggest_int(*NUM_EPOCHS)
    num_layers = trial.suggest_int(*NUM_LAYERS)
    dropout_rate = trial.suggest_float(*DROPOUT_RATE)
    lr = trial.suggest_loguniform(*LR)
    activation_fn = ACTIVATION_FN

    print(f"hidden_dim: {hidden_dim}, batch_size: {batch_size}, num_epochs: {num_epochs}, num_layers: {num_layers}, dropout_rate: {dropout_rate}, lr: {lr}, activation_fn: {activation_fn}")

        
    # Lists to store metrics for each fold
    all_fold_accuracies = []
    all_fold_precisions = []
    all_fold_recalls = []
    all_fold_f1s = []


    output_dim = 1
    k_folds = NUM_FOLDS
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

        smote = SMOTE(sampling_strategy="auto", random_state=SEED)
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