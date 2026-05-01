import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

os.makedirs("outputs", exist_ok=True)
os.makedirs("demo_files", exist_ok=True)

llama_train = pd.read_csv("outputs/features_Meta_Llama_3_8B_Instruct_train.csv")
llama_val = pd.read_csv("outputs/features_Meta_Llama_3_8B_Instruct_validation.csv")

mistral_train = pd.read_csv("outputs/features_Mistral_7B_Instruct_v0_3_train.csv")
mistral_val = pd.read_csv("outputs/features_Mistral_7B_Instruct_v0_3_validation.csv")

gemma_train = pd.read_csv("outputs/features_gemma_2_9b_it_train.csv")
gemma_val = pd.read_csv("outputs/features_gemma_2_9b_it_validation.csv")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

FEATURES = [
    "entropy",
    "margin",
    "length_norm_logprob",
    "selected_num_tokens",
]

def prepare_xy(df):
    df = df.copy()

    y = 1 - df["correct"].astype(int)
    X = df[FEATURES]

    return X.to_numpy(), y.to_numpy()

def compute_metrics(y_true, logits):
    probs = torch.sigmoid(torch.as_tensor(logits)).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "auroc": roc_auc_score(y_true, probs),
    }

# Train MLP and save checkpoint
def train_small_mlp(train_df, val_df, checkpoint_path=None, batch_size=32, epochs=100, lr=3e-4,):
    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=min(batch_size, len(X_train_t)),
        shuffle=True,
    )

    model = MLP(input_dim=X_train.shape[1]).to(DEVICE)

    pos_count = max(int(y_train.sum()), 1)
    neg_count = max(int((1 - y_train).sum()), 1)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val_auroc = float("-inf")
    best_epoch = -1
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val_t.to(DEVICE)).cpu().numpy()

        val_metrics = compute_metrics(y_val, val_logits)
        avg_train_loss = epoch_loss / len(X_train_t)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    if checkpoint_path is not None:
        torch.save({
            "model": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "features": FEATURES,
            "best_epoch": best_epoch,
            "best_val_auroc": best_val_auroc,
            "input_dim": X_train.shape[1],
        }, checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")

    with torch.no_grad():
        final_val_logits = model(X_val_t.to(DEVICE)).cpu().numpy()

    final_metrics = compute_metrics(y_val, final_val_logits)

    return {
        "model": model,
        "scaler": scaler,
        "results": final_metrics,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
    }

# Run all combinations of train to test
experiments = [
    ("llama", "llama", llama_train, llama_val),
    ("llama", "mistral", llama_train, mistral_val),
    ("llama", "gemma", llama_train, gemma_val),
    ("mistral", "mistral", mistral_train, mistral_val),
    ("mistral", "llama", mistral_train, llama_val),
    ("mistral", "gemma", mistral_train, gemma_val),
    ("gemma", "gemma", gemma_train, gemma_val),
    ("gemma", "mistral", gemma_train, mistral_val),
    ("gemma", "llama", gemma_train, llama_val),
]

rows = []

for train_model, test_model, train_df, val_df in experiments:
    checkpoint_path = f"demo_files/mlp_checkpoint_{train_model}_to_{test_model}.pt"

    artifacts = train_small_mlp(
        train_df,
        val_df,
        checkpoint_path=checkpoint_path,
    )

    row = {
        "features": "multi_signal_mlp",
        "train_model": train_model,
        "test_model": test_model,
        "best_epoch": artifacts["best_epoch"],
        "checkpoint_path": checkpoint_path,
        **artifacts["results"],
    }
    rows.append(row)

    print(f"\nTrain: {train_model} | Test: {test_model}")
    print(f"Best epoch: {artifacts['best_epoch']}")
    print(artifacts["results"])
    print(artifacts["history"].tail())

# Run combined experiments
combined_train = pd.read_csv("outputs/features_combined_train.csv")

pooled_experiments = [
    ("combined", "llama", combined_train, llama_val),
    ("combined", "mistral", combined_train, mistral_val),
    ("combined", "gemma", combined_train, gemma_val),
]

for train_model, test_model, train_df, val_df in pooled_experiments:
    checkpoint_path = f"demo_files/mlp_checkpoint_{train_model}_to_{test_model}.pt"

    artifacts = train_small_mlp(
        train_df,
        val_df,
        checkpoint_path=checkpoint_path,
    )

    row = {
        "features": "multi_signal_mlp",
        "train_model": train_model,
        "test_model": test_model,
        "best_epoch": artifacts["best_epoch"],
        "checkpoint_path": checkpoint_path,
        **artifacts["results"],
    }
    rows.append(row)

    print(f"\nTrain: {train_model} | Test: {test_model}")
    print(f"Best epoch: {artifacts['best_epoch']}")
    print(artifacts["results"])
    print(artifacts["history"].tail())


results_df = pd.DataFrame(rows)
out_path = "outputs/mlp_results.csv"
results_df.to_csv(out_path, index=False)

print(f"\nSaved MLP results to {out_path}")
print(results_df.to_string(index=False))