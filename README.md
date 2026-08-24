# EEG-LoRE: Language-Oriented Routing of Experts for Unified Multi-Task EEG Decoding

This repository contains the training and direct-inference implementation of EEG-LoRE.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place the 16 processed datasets under one directory. Each subject-oriented HDF5 file uses one group per subject with datasets `X` and `Y`. The four SEED files and the ADHD file use `trainX`, `trainY`, `validX`, `validY`, `testX`, and `testY`. EEG arrays have shape `(trials, 65, samples)`, use a sampling rate of 200 Hz, and contain integer labels beginning at zero.

The required filenames are listed in `eeg_lore/metadata.py` with the `.h5` suffix.

Pretraining data consist of NumPy arrays with shape `(trials, 65, samples)`.

## Pretraining

```bash
python pretrain.py --data-root path/to/pretraining_arrays --output outputs/pretraining --gpu 0
```

## Multi-task tuning

```bash
python train.py --stage a --data-root path/to/downstream_data --init-checkpoint outputs/pretraining/best.pt --output outputs/stage_a --gpu 0
python train.py --stage b --data-root path/to/downstream_data --init-checkpoint outputs/stage_a/best.pt --output outputs/stage_b --gpu 0
```

## Direct inference

```bash
python evaluate.py --data-root path/to/downstream_data --checkpoint checkpoints/eeg_lore_seed42.pt --gpu 0
```

The released checkpoint is the final instruction-tuned EEG-LoRE model initialized from the self-supervised pretraining stage.
