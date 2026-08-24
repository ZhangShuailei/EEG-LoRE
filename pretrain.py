import argparse
import random
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from eeg_lore.config import ModelConfig
from eeg_lore.model import EEGEncoder
from eeg_lore.utils import cosine_schedule, save_state, seed_everything


class PretrainingDataset(Dataset):
    def __init__(self, root, keys, training, length=2000):
        self.root = Path(root)
        self.keys = keys
        self.training = training
        self.length = length
        self.arrays = {}

    def __len__(self):
        return len(self.keys)

    def array(self, name):
        if name not in self.arrays:
            self.arrays[name] = np.load(self.root / f"{name}.npy", mmap_mode="r")
        return self.arrays[name]

    def __getitem__(self, index):
        name, item = self.keys[index]
        sample = np.asarray(self.array(name)[item]).copy()
        if sample.shape[1] < self.length:
            missing = self.length - sample.shape[1]
            left = random.randint(0, missing) if self.training else 0
            sample = np.pad(sample, ((0, 0), (left, missing - left)), mode="constant")
        elif sample.shape[1] > self.length:
            start = random.randint(0, sample.shape[1] - self.length) if self.training else 0
            sample = sample[:, start:start + self.length]
        return np.ascontiguousarray(sample, dtype=np.float32)


def split_files(root, validation_ratio=0.1):
    interval = int(1.0 / validation_ratio)
    train, valid = [], []
    counter = 0
    for path in sorted(Path(root).glob("*.npy")):
        size = np.load(path, mmap_mode="r").shape[0]
        for index in range(size):
            target = valid if counter % interval == interval - 1 else train
            target.append((path.stem, index))
            counter += 1
    return train, valid


class PretrainingModule(pl.LightningModule):
    def __init__(self, config, output, steps_per_epoch, learning_rate, epochs, warmup_epochs):
        super().__init__()
        self.model = EEGEncoder(config)
        self.output = Path(output)
        self.learning_rate = learning_rate
        self.schedule = cosine_schedule(1.0, 0.1, epochs, steps_per_epoch, warmup_epochs)
        self.best_loss = float("inf")
        self.history = []

    def configure_optimizers(self):
        fast = list(self.model.tokenizer.parameters())
        fast_ids = {id(parameter) for parameter in fast}
        slow = [parameter for parameter in self.model.parameters() if id(parameter) not in fast_ids]
        optimizer = optim.AdamW([
            {"params": fast, "lr": self.learning_rate},
            {"params": slow, "lr": self.learning_rate * 0.1},
        ])
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: self.schedule[min(max(step - 1, 0), len(self.schedule) - 1)],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_index):
        return self.model.reconstruction_loss(batch)

    def validation_step(self, batch, batch_index):
        loss = self.model.reconstruction_loss(batch)
        self.log("validation_loss", loss, on_epoch=True)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or self.global_rank != 0:
            return
        loss = float(self.trainer.callback_metrics["validation_loss"])
        self.history.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            save_state(self.model, self.output / "best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    train_keys, valid_keys = split_files(args.data_root)
    train_dataset = PretrainingDataset(args.data_root, train_keys, True)
    valid_dataset = PretrainingDataset(args.data_root, valid_keys, False)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
    args.output.mkdir(parents=True, exist_ok=True)
    module = PretrainingModule(ModelConfig(), args.output, len(train_loader), args.learning_rate, args.epochs, args.warmup_epochs)
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
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if trainer.global_rank == 0:
        np.save(args.output / "validation.npy", np.asarray(module.history, dtype=np.float64))


if __name__ == "__main__":
    main()
