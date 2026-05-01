import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Main split is about 9.7k train / 1.2k validation.
# The test split does not include labels, so only train/validation are used here.
ds = load_dataset("tau/commonsense_qa")
print(ds)
print(ds["train"][0])

# Format Commonsense questions
def normalize_csqa_example(example):
    labels = list(example["choices"]["label"])
    texts = list(example["choices"]["text"])

    label_to_idx = {label: i for i, label in enumerate(labels)}
    gold_idx = label_to_idx[example["answerKey"]]

    return {
        "id": example.get("id", None),
        "question": example["question"],
        "question_concept": example.get("question_concept", None),
        "choice_labels": labels,
        "choice_texts": texts,
        "gold_idx": gold_idx,
        "gold_label": example["answerKey"],
    }


sample = normalize_csqa_example(ds["train"][0])
print(sample)

# Load models in 4-bit for memory consistency
def load_model_and_tokenizer(model_name, dtype=torch.float16, load_in_4bit=True, attn_implementation=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
    }

    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    return tokenizer, model


def format_prompt(question, choice_labels, choice_texts):
    options_block = "\n".join(
        f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)
    )

    prompt = (
        "You are answering a multiple-choice commonsense question.\n"
        "Choose the single best answer.\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_block}\n\n"
        "Answer:"
    )

    return prompt

# Find likelihood of answers to score later
def find_answer_logprobs(model, tokenizer, prompt, text):
    device = model.device

    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    text_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    input_ids = torch.cat([prompt_ids, text_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    prompt_len = prompt_ids.shape[1]
    text_len = text_ids.shape[1]

    text_token_log_probs = token_log_probs[:, prompt_len - 1 : prompt_len - 1 + text_len]

    total_logprob = text_token_log_probs.sum().item()
    avg_logprob = text_token_log_probs.mean().item()
    nll = -total_logprob

    return {
        "total_logprob": total_logprob,
        "avg_logprob": avg_logprob,
        "nll": nll,
        "num_tokens": int(text_len),
    }

def score_answer_options(model, tokenizer, question, choice_labels, choice_texts):
    prompt = format_prompt(question, choice_labels, choice_texts)

    option_scores = []
    for label, answer_text in zip(choice_labels, choice_texts):
        scored_text = f" {answer_text}"
        score = find_answer_logprobs(model, tokenizer, prompt, scored_text)
        option_scores.append({
            "label": label,
            "text": answer_text,
            **score,
        })

    avg_logprobs = torch.tensor([x["avg_logprob"] for x in option_scores], dtype=torch.float32)
    probs = torch.softmax(avg_logprobs, dim=0).numpy()

    for i, p in enumerate(probs):
        option_scores[i]["prob"] = float(p)

    pred_idx = int(np.argmax(probs))
    return {
        "prompt": prompt,
        "option_scores": option_scores,
        "pred_idx": pred_idx,
        "pred_label": choice_labels[pred_idx],
        "pred_text": choice_texts[pred_idx],
        "prob_vector": probs.tolist(),
    }

def extract_features(prob_vector, option_scores, pred_idx):
    p = np.array(prob_vector, dtype=np.float64)
    p = p / p.sum()

    entropy = float(-(p * np.log(p + 1e-12)).sum())

    sorted_p = np.sort(p)[::-1]
    margin = float(sorted_p[0] - sorted_p[1])

    selected = option_scores[pred_idx]
    length_norm_logprob = float(selected["avg_logprob"])
    selected_total_logprob = float(selected["total_logprob"])
    selected_num_tokens = int(selected["num_tokens"])

    return {
        "entropy": entropy,
        "margin": margin,
        "length_norm_logprob": length_norm_logprob,
        "selected_total_logprob": selected_total_logprob,
        "selected_num_tokens": selected_num_tokens,
    }

def build_feature_dataframe(dataset_split, model, tokenizer, max_examples=None, split_name="validation"):
    rows = []
    n = len(dataset_split) if max_examples is None else min(max_examples, len(dataset_split))

    for i in tqdm(range(n), desc=f"Scoring {split_name}"):
        ex = normalize_csqa_example(dataset_split[i])
        result = score_answer_options(
            model,
            tokenizer,
            ex["question"],
            ex["choice_labels"],
            ex["choice_texts"],
        )
        feats = extract_features(result["prob_vector"], result["option_scores"], result["pred_idx"])

        rows.append({
            "row_idx": i,
            "id": ex["id"],
            "split": split_name,
            "question": ex["question"],
            "question_concept": ex["question_concept"],
            "gold_idx": ex["gold_idx"],
            "gold_label": ex["gold_label"],
            "pred_idx": result["pred_idx"],
            "pred_label": result["pred_label"],
            "correct": int(result["pred_idx"] == ex["gold_idx"]),
            **feats,
            "prob_vector": json.dumps(result["prob_vector"]),
            "choice_labels": json.dumps(ex["choice_labels"]),
            "choice_texts": json.dumps(ex["choice_texts"]),
        })

    return pd.DataFrame(rows)

# Models and inference loop
MODEL_CONFIGS = [
    {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "load_in_4bit": True,
    },
    {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "load_in_4bit": True,
    },
    {
        "model_name": "google/gemma-2-9b-it",
        "load_in_4bit": True,
    },
]

# SPLITS = {
#     "train": ds["train"],
#     "validation": ds["validation"],
# }

SMOKE_TEST = True
SMOKE_N = 100

SPLITS = {
    "train": ds["train"].select(range(SMOKE_N)) if SMOKE_TEST else ds["train"],
    "validation": ds["validation"].select(range(SMOKE_N)) if SMOKE_TEST else ds["validation"],
}

for cfg in MODEL_CONFIGS:
    model_name = cfg["model_name"]

    print(f"Loading model: {model_name}")
    print(f"{'=' * 80}")

    tokenizer, model = load_model_and_tokenizer(**cfg)

    name = model_name.split("/")[-1].replace(".", "_").replace("-", "_")

    for split_name, split_ds in SPLITS.items():
        print(f"\nRunning inference for {model_name} on {split_name}...")

        df_split = build_feature_dataframe(
            split_ds,
            model,
            tokenizer,
            max_examples=None,
            split_name=split_name,
        )

        df_split["model_name"] = model_name

        out_path = OUTPUT_DIR / f"features_{name}_{split_name}.csv"
        df_split.to_csv(out_path, index=False)

        print(f"Saved {len(df_split)} rows to {out_path}")

        del df_split

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


print("LLAMA STATS")
llama_df_train = pd.read_csv("outputs/features_Meta_Llama_3_8B_Instruct_train.csv")
print(llama_df_train["correct"].value_counts())

llama_df_val = pd.read_csv("outputs/features_Meta_Llama_3_8B_Instruct_validation.csv")
print(llama_df_val["correct"].value_counts())

print("MISTRAL STATS")
mistral_df_train = pd.read_csv("outputs/features_Mistral_7B_Instruct_v0_3_train.csv")
print(mistral_df_train["correct"].value_counts())

mistral_df_val = pd.read_csv("outputs/features_Mistral_7B_Instruct_v0_3_validation.csv")
print(mistral_df_val["correct"].value_counts())

print("GEMMA STATS")
gemma_df_train = pd.read_csv("outputs/features_gemma_2_9b_it_train.csv")
print(gemma_df_train["correct"].value_counts())

gemma_df_val = pd.read_csv("outputs/features_gemma_2_9b_it_validation.csv")
print(gemma_df_val["correct"].value_counts())

combined_train_paths = [
    "outputs/features_Meta_Llama_3_8B_Instruct_train.csv",
    "outputs/features_Mistral_7B_Instruct_v0_3_train.csv",
    "outputs/features_gemma_2_9b_it_train.csv",
]

combined_train_dfs = []

for path in combined_train_paths:
    df = pd.read_csv(path)
    combined_train_dfs.append(df)

combined_train = pd.concat(combined_train_dfs, ignore_index=True)

combined_out_path = "outputs/features_combined_train.csv"
combined_train.to_csv(combined_out_path, index=False)

print(f"Saved combined train features to {combined_out_path}")
print(f"Combined train rows: {len(combined_train)}")
