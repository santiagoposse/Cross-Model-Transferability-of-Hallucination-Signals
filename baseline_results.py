from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


OUTPUT_DIR = Path("./outputs")

FEATURE_SETS = {
    "entropy_only": ["entropy"],
    "margin_only": ["margin"],
    "logprob_only": ["length_norm_logprob"],
    "tokens_only": ["selected_num_tokens"],
}

def prepare_xy(df, features):
    df = df.copy()
    
    y = 1 - df["correct"].astype(int)
    X = df[features]

    return X, y

# Baseline experiments 
def train_and_evaluate(train_df, test_df, features):
    X_train, y_train = prepare_xy(train_df, features)
    X_test, y_test = prepare_xy(test_df, features)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auroc": roc_auc_score(y_test, y_prob),
    }

    return clf, results

model_files = {
    "llama": {
        "train": OUTPUT_DIR / "features_Meta_Llama_3_8B_Instruct_train.csv",
        "validation": OUTPUT_DIR / "features_Meta_Llama_3_8B_Instruct_validation.csv",
    },
    "mistral": {
        "train": OUTPUT_DIR / "features_Mistral_7B_Instruct_v0_3_train.csv",
        "validation": OUTPUT_DIR / "features_Mistral_7B_Instruct_v0_3_validation.csv",
    },
    "gemma": {
        "train": OUTPUT_DIR / "features_gemma_2_9b_it_train.csv",
        "validation": OUTPUT_DIR / "features_gemma_2_9b_it_validation.csv",
    },
}

dfs = {
    model_name: {
        split: pd.read_csv(path)
        for split, path in split_dict.items()
    }
    for model_name, split_dict in model_files.items()
}

rows = []

for baseline_name, features in FEATURE_SETS.items():
    for train_model in dfs:
        for test_model in dfs:
            _, metrics = train_and_evaluate(
                dfs[train_model]["train"],
                dfs[test_model]["validation"],
                features,
            )

            rows.append({
                "baseline": baseline_name,
                "features": ", ".join(features),
                "train_model": train_model,
                "test_model": test_model,
                **metrics,
            })

results_df = pd.DataFrame(rows)

out_path = OUTPUT_DIR / "baseline_results.csv"
results_df.to_csv(out_path, index=False)

print(f"Saved baseline results to {out_path}")
print("\nEntropy baseline")
print(results_df[results_df["baseline"] == "entropy_only"].to_string(index=False))

print("\nLength-normalized log-probability baseline")
print(results_df[results_df["baseline"] == "logprob_only"].to_string(index=False))

print("\nToken-count baseline")
print(results_df[results_df["baseline"] == "tokens_only"].to_string(index=False))  

print("\nTop2 margin baseline")
print(results_df[results_df["baseline"] == "margin_only"].to_string(index=False))  
