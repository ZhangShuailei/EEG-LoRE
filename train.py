import argparse
import random
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from eeg_lore.config import ModelConfig
from eeg_lore.data import load_dataset
from eeg_lore.metadata import DATASETS, DATASET_IDS, DATASET_TO_TASK, INSTRUCTIONS, TARGETS, TASK_DESCRIPTIONS, TASK_NAMES, load_text_assets
from eeg_lore.model import EEGLoRE, EEGLoREBase
from eeg_lore.utils import cosine_schedule, save_state, seed_everything


class TuningDataset(Dataset):
    def __init__(self, keys, signals, labels, subjects, label_indices, text_embeddings, training, subject_indices, default_probability):
        self.keys = keys
        self.signals = signals
        self.labels = labels
        self.subjects = subjects
        self.label_indices = label_indices
        self.text_embeddings = text_embeddings
        self.training = training
        self.subject_indices = subject_indices
        self.default_probability = default_probability

    def __len__(self):
        return len(self.keys)

    def fit_length(self, sample, length=2000):
        if sample.shape[1] < length:
            missing = length - sample.shape[1]
            left = random.randint(0, missing) if self.training else 0
            sample = np.pad(sample, ((0, 0), (left, missing - left)), mode="constant")
        elif sample.shape[1] > length:
            start = random.randint(0, sample.shape[1] - length) if self.training else 0
            sample = sample[:, start:start + length]
        return np.ascontiguousarray(sample, dtype=np.float32)

    def __getitem__(self, index):
        dataset, local_index = self.keys[index]
        if random.random() < self.default_probability:
            text = random.choice(INSTRUCTIONS["Default"])
        else:
            text = random.choice(INSTRUCTIONS[dataset])
        subject = int(self.subjects[dataset][local_index])
        subject_id = -1 if subject < 0 else self.subject_indices[(dataset, subject)]
        return (
            self.fit_length(self.signals[dataset][local_index]),
            self.text_embeddings[text],
            self.label_indices[self.labels[dataset][local_index]],
            DATASET_IDS[dataset],
            subject_id,
        )


def load_training_data(root):
    train_keys, valid_keys = [], []
    train_x, train_y, train_s = {}, {}, {}
    valid_x, valid_y, valid_s = {}, {}, {}
    for dataset in DATASETS:
        split = load_dataset(dataset, root)
        train_x[dataset] = split.train_x
        train_y[dataset] = [TARGETS[dataset][int(value)] for value in split.train_y]
        train_s[dataset] = split.train_subjects
        valid_x[dataset] = split.valid_x
        valid_y[dataset] = [TARGETS[dataset][int(value)] for value in split.valid_y]
        valid_s[dataset] = split.valid_subjects
        train_keys.extend((dataset, index) for index in range(len(split.train_x)))
        valid_keys.extend((dataset, index) for index in range(len(split.valid_x)))
    return (train_keys, train_x, train_y, train_s), (valid_keys, valid_x, valid_y, valid_s)


def subject_map(train_subjects, valid_subjects):
    mapping = {}
    for dataset in DATASETS:
        values = np.concatenate((train_subjects[dataset], valid_subjects[dataset]))
        for subject in np.unique(values):
            subject = int(subject)
            if subject >= 0 and (dataset, subject) not in mapping:
                mapping[(dataset, subject)] = len(mapping)
    return mapping


def checkpoint_state(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {key.removeprefix("model."): value for key, value in state.items()}


class TuningModule(pl.LightningModule):
    def __init__(self, stage, config, prototypes, task_embeddings, init_checkpoint, output, steps_per_epoch, subject_count, seed, learning_rate, epochs, warmup_epochs, weight_decay, adapter_regularization):
        super().__init__()
        seed_everything(seed)
        model_class = EEGLoREBase if stage == "a" else EEGLoRE
        self.model = model_class(
            config,
            DATASET_TO_TASK,
            len(TASK_NAMES),
            len(DATASETS),
            prototypes,
            task_embeddings,
            config.dropout,
        )
        self.stage = stage
        self.subject_classifier = nn.Sequential(
            nn.Linear(config.dim_text_emb, config.dim_text_emb),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.dim_text_emb, subject_count),
        )
        self.output = Path(output)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adapter_regularization = adapter_regularization
        self.schedule = cosine_schedule(1.0, 0.1, epochs, steps_per_epoch, warmup_epochs)
        self.history = []
        self.best_accuracy = -1.0
        state = checkpoint_state(init_checkpoint)
        state.pop("prototypes", None)
        state.pop("meta_embs", None)
        model_state = self.model.state_dict()
        mapped = {}
        for key, value in state.items():
            if key in model_state:
                mapped[key] = value
            elif f"tower.{key}" in model_state:
                mapped[f"tower.{key}"] = value
        compatible = {key: value for key, value in mapped.items() if value.shape == model_state[key].shape}
        self.model.load_state_dict(compatible, strict=False)

    def configure_optimizers(self):
        expert_parameters = list(self.model.task_adaln.parameters()) + list(self.model.task_proj.parameters())
        fast_parameters = list(self.model.tower.tokenizer.parameters()) + list(self.model.ds_embed.parameters()) + [self.model.ds_scale] + expert_parameters + list(self.subject_classifier.parameters())
        adapter_parameters = list(self.model.proto_adapters.parameters()) if self.stage == "b" else []
        fast_ids = {id(parameter) for parameter in fast_parameters + adapter_parameters}
        slow_parameters = [parameter for parameter in self.model.parameters() if id(parameter) not in fast_ids]
        groups = [{"params": fast_parameters, "lr": self.learning_rate}]
        if adapter_parameters:
            groups.append({"params": adapter_parameters, "lr": self.learning_rate})
        groups.append({"params": slow_parameters, "lr": self.learning_rate * 0.1})
        optimizer = optim.AdamW(groups, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: self.schedule[min(max(step - 1, 0), len(self.schedule) - 1)],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_index):
        x, instruction_embedding, labels, dataset_ids, subjects = batch
        embedding = self.model(x, instruction_embedding, dataset_ids)[0]
        if self.stage == "a":
            logits = embedding @ self.model.prototypes
            loss = F.cross_entropy(logits, labels)
        else:
            logits = self.model.logits_from_embedding(embedding, dataset_ids)
            loss = F.cross_entropy(logits, labels) + self.adapter_regularization * self.model.adapter_loss()
        return loss

    def validation_step(self, batch, batch_index):
        x, instruction_embedding, labels, dataset_ids, subjects = batch
        logits = self.model.get_logits(x, instruction_embedding, dataset_ids)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or self.global_rank != 0:
            return
        loss = float(self.trainer.callback_metrics["validation_loss"])
        accuracy = float(self.trainer.callback_metrics["validation_accuracy"])
        self.history.append((loss, accuracy))
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            save_state(self.model, self.output / "best.pt")

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking or self.global_rank != 0:
            return
        epoch = self.current_epoch + 1
        if epoch % 10 == 0:
            save_state(self.model, self.output / f"epoch_{epoch:03d}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--text-assets", type=Path, default=Path("assets/text_embeddings"))
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--adapter-regularization", type=float, default=0.1)
    parser.add_argument("--default-instruction-probability", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="32-true")
    args = parser.parse_args()
    config = ModelConfig()
    prototypes, label_indices, text_embeddings = load_text_assets(args.text_assets)
    task_embeddings = np.asarray([text_embeddings[text] for text in TASK_DESCRIPTIONS], dtype=np.float32)
    train_data, valid_data = load_training_data(args.data_root)
    subjects = subject_map(train_data[3], valid_data[3])
    train_dataset = TuningDataset(*train_data, label_indices, text_embeddings, True, subjects, args.default_instruction_probability)
    valid_dataset = TuningDataset(*valid_data, label_indices, text_embeddings, False, subjects, args.default_instruction_probability)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    args.output.mkdir(parents=True, exist_ok=True)
    module = TuningModule(
        args.stage,
        config,
        prototypes,
        task_embeddings,
        args.init_checkpoint,
        args.output,
        len(train_loader),
        len(subjects),
        args.seed,
        args.learning_rate,
        args.epochs,
        args.warmup_epochs,
        args.weight_decay,
        args.adapter_regularization,
    )
    devices = [int(value) for value in args.gpu.split(",")]
    precision = int(args.precision) if args.precision.isdigit() else args.precision
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices if torch.cuda.is_available() else 1,
        max_epochs=args.epochs,
        precision=precision,
        enable_checkpointing=False,
        benchmark=True,
        deterministic=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if trainer.global_rank == 0:
        np.save(args.output / "validation.npy", np.asarray(module.history, dtype=np.float64))


if __name__ == "__main__":
    main()
