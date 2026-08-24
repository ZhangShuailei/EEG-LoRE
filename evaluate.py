import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from eeg_lore.config import ModelConfig
from eeg_lore.data import load_dataset
from eeg_lore.metadata import DATASETS, DATASET_IDS, DATASET_TO_TASK, TARGETS, TASK_DESCRIPTIONS, TASK_NAMES, instruction_text, load_text_assets
from eeg_lore.model import EEGLoRE


@torch.no_grad()
def evaluate_dataset(model, loader, device, instruction_embedding, dataset_id, class_indices):
    logits_all, targets_all = [], []
    model.eval()
    for x, targets in loader:
        x = x.to(device, non_blocking=True)
        dataset_ids = torch.full((x.size(0),), dataset_id, device=device, dtype=torch.long)
        logits = model.direct_logits(x, instruction_embedding, dataset_ids)[:, class_indices]
        logits_all.append(logits.float().cpu())
        targets_all.append(targets)
    logits = torch.cat(logits_all).numpy()
    targets = torch.cat(targets_all).numpy()
    predictions = logits.argmax(axis=-1)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    weighted_f1 = f1_score(targets, predictions, average="weighted")
    kappa = cohen_kappa_score(targets, predictions)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    try:
        if probabilities.shape[1] == 2:
            auroc = roc_auc_score(targets, probabilities[:, 1])
        else:
            auroc = roc_auc_score(targets, probabilities, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")
    return balanced_accuracy, auroc, weighted_f1, kappa


def checkpoint_state(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {key.removeprefix("model."): value for key, value in state.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--text-assets", type=Path, default=Path("assets/text_embeddings"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/eeg_lore_seed42.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/direct_inference.tsv"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    prototypes, label_indices, text_embeddings = load_text_assets(args.text_assets)
    task_embeddings = np.asarray([text_embeddings[text] for text in TASK_DESCRIPTIONS], dtype=np.float32)
    model = EEGLoRE(
        ModelConfig(),
        DATASET_TO_TASK,
        len(TASK_NAMES),
        len(DATASETS),
        prototypes,
        task_embeddings,
    ).to(device)
    model.load_state_dict(checkpoint_state(args.checkpoint), strict=True)
    rows = []
    for dataset in args.datasets:
        split = load_dataset(dataset, args.data_root)
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(split.test_x, dtype=torch.float32),
                torch.as_tensor(split.test_y, dtype=torch.long),
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        text = instruction_text(dataset, "specific")
        instruction_embedding = torch.as_tensor(text_embeddings[text], device=device)
        class_indices = [label_indices[label] for label in TARGETS[dataset]]
        metrics = evaluate_dataset(
            model,
            loader,
            device,
            instruction_embedding,
            DATASET_IDS[dataset],
            class_indices,
        )
        rows.append((dataset, *metrics))
        print(dataset, *(f"{value:.6f}" for value in metrics))
    average_bacc = float(np.mean([row[1] for row in rows]))
    average_kappa = float(np.mean([row[4] for row in rows]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write("dataset\tbacc\tauroc\twf1\tkappa\n")
        for dataset, bacc, auroc, wf1, kappa in rows:
            handle.write(f"{dataset}\t{bacc:.6f}\t{auroc:.6f}\t{wf1:.6f}\t{kappa:.6f}\n")
        handle.write(f"AVERAGE\t{average_bacc:.6f}\t\t\t{average_kappa:.6f}\n")
    print("AVERAGE", f"{average_bacc:.6f}", f"{average_kappa:.6f}")


if __name__ == "__main__":
    main()
