# Credit Card Fraud Detection ML

This project implements a Graph Neural Network (GNN) approach for credit card fraud detection, where transactions are modeled as edges in a heterogeneous graph connecting users and merchants.  
The system performs edge-level binary classification (fraud vs. non-fraud) and uses Optuna for automated hyperparameter optimization with K-Fold cross-validation.


## Key Features

- Graph-based representation of credit card transactions
- GNN architectures with attention and neural message passing
- SMOTE for handling severe class imbalance
- Optuna for hyperparameter optimization
- K-Fold cross-validation with F1-score optimization
- GPU support


## Problem Formulation

- **Nodes**
  - Users
  - Merchants

- **Edges**
  - Individual credit card transactions

- **Edge Features**
  - Transaction amount
  - Time (hour, minute, day, month, year)
  - Merchant category code (MCC)
  - Card type
  - Chip usage
  - Error flags

- **Task**
  - Binary edge classification:  
    **Fraud (1)** vs **Legitimate (0)**

## Model Architecture

The model performs edge classification using node embeddings produced by a GNN.

### Core Components

- **GNN Layers**
  - `GATConv` or `NNConv` (edge-conditioned message passing)
- **Edge Representation**
  - Concatenation of source and destination node embeddings
- **Output**
  - Sigmoid-activated binary prediction per edge

## Class Imbalance Handling

Credit card fraud data is highly imbalanced.

This project addresses imbalance by:
1. Standardizing edge features
2. Applying SMOTE on edge attributes
3. Generating synthetic edges with randomized valid node pairs
4. Injecting synthetic samples into the training graph

This is performed inside each training fold to avoid data leakage.


## Hyperparameter Optimization

Hyperparameters are optimized using Optuna, targeting maximum average F1-score across K folds.

### Tuned Hyperparameters

- Hidden dimension size
- Batch size
- Number of GNN layers
- Number of epochs
- Dropout rate
- Learning rate

### Optimization Strategy

- Objective: maximize F1-score
- Evaluation: K-Fold Cross-Validation
- Best model saved per fold


## Training Pipeline

1. Load and preprocess transaction data
2. Build graph representation
3. Run Optuna optimization
4. Train final model using best hyperparameters
5. Evaluate with K-Fold cross-validation
6. Save trained models


## Evaluation Metrics

For each fold, the following metrics are reported:
- Accuracy
- Precision
- Recall
- F1-score (primary optimization metric)

Final performance is reported as the average across folds.


## Project Structure

```bash
.
├── main.py                         # Entry point
├── optimization.py                 # Optuna objective and CV logic
├── model.py                        # GNN edge classifier
├── dataset.py                      # PyG Dataset wrapper
├── graph.py                        # Graph construction logic
├── data.py                         # Data preprocessing
├── config.py                       # Hyperparameters and constants
├── requirements.txt
└── README.md
```


## Running the Project

### Clone the Repository
```bash
git clone https://github.com/fabiolariveranoriega/cc-fraud-detection-ml
cd cc-fraud-detection-ml
```

### Setup Environment

Create and activate a virtual environment (recommended), then install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Run the training pipeline
```bash
python main.py --path path/to/credit_card_transactions.csv
```

Arguments:
- `--path`: Path to CSV file containing credit card fraud data

The script will:

1. Preprocess the raw transaction data
2. Construct a graph of users and merchants
3. Run Optuna to find optimal hyperparameters
4. Train the final GNN model using K-Fold cross-validation
5. Save trained models for each fold to disk