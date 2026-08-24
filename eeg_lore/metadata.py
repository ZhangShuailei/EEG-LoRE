from pathlib import Path

import numpy as np


INSTRUCTIONS = {
    "Default": ["None", "Default"],
    "MI_OpenBMI": ["Decode motor imagery", "Decode (Left vs Right) hand motor imagery"],
    "MI_BCIC_IV2a": ["Decode motor imagery", "Decode (Left vs Right vs Foot vs Tongue) motor imagery"],
    "MI_BCIC_Upperlimb": ["Decode motor imagery", "Decode (Cylindrical, Spherical, Lumbrical) hand movements"],
    "MI_ShanghaiU": ["Decode motor imagery", "Decode (Left vs Right) hand motor imagery"],
    "MI_HighGamma": ["Decode motor imagery", "Decode (Left vs Right vs Foot) motor imagery"],
    "MI_Cho2017": ["Decode motor imagery", "Decode (Left vs Right) hand motor imagery"],
    "MI_Shin2017A": ["Decode motor imagery", "Decode (Left vs Right) hand motor imagery"],
    "MI_PhysioNet": ["Decode motor imagery", "Decode (Left vs Right) hand motor imagery"],
    "EMO_FACED": ["Decode emotional states", "Decode emotional states (Anger, Fear, Disgust, Sad, Amusement, Inspiration, Joy, Tenderness, Neutral)"],
    "EMO_SEED_3_seg4": ["Decode emotional states", "Decode emotional states (Negative, Neutral, Positive)"],
    "EMO_SEED_4_seg4": ["Decode emotional states", "Decode emotional states (Neutral, Sad, Fear, Happy)"],
    "EMO_SEED_5_seg4": ["Decode emotional states", "Decode emotional states (Disgust, Fear, Sad, Neutral, Happy)"],
    "EMO_SEED_7_seg4": ["Decode emotional states", "Decode emotional states (Happy, Surprise, Neutral, Sad, Disgust, Fear, Anger)"],
    "CS_BCIC_Speech": ["Decode covert speech", "Decode covert speech (hello, help-me, stop, thank-you, yes)"],
    "ADHD_AliMotie": ["Decode mental disorder", "Decode ADHD vs Healthy"],
    "Workload": ["Decode mental workload states", "Decode mental workload states (Resting vs Workload)"],
}

TARGETS = {
    "MI_OpenBMI": ["Right", "Left"],
    "MI_BCIC_IV2a": ["Left", "Right", "Foot", "Tongue"],
    "MI_BCIC_Upperlimb": ["Cylin", "Sphe", "Lumbrical"],
    "MI_ShanghaiU": ["Left", "Right"],
    "MI_HighGamma": ["Right", "Left", "Foot"],
    "MI_Cho2017": ["Left", "Right"],
    "MI_Shin2017A": ["Left", "Right"],
    "MI_PhysioNet": ["Left", "Right"],
    "EMO_FACED": ["Anger", "Fear", "Disgust", "Sad", "Amusement", "Inspiration", "Joy", "Tenderness", "Neutral"],
    "EMO_SEED_3_seg4": ["Negative", "Neutral", "Positive"],
    "EMO_SEED_4_seg4": ["Neutral", "Sad", "Fear", "Happy"],
    "EMO_SEED_5_seg4": ["Disgust", "Fear", "Sad", "Neutral", "Happy"],
    "EMO_SEED_7_seg4": ["Happy", "Surprise", "Neutral", "Sad", "Disgust", "Fear", "Anger"],
    "CS_BCIC_Speech": ["hello", "help-me", "stop", "thank-you", "yes"],
    "ADHD_AliMotie": ["Healthy", "ADHD"],
    "Workload": ["Resting", "Workload"],
}

DATASETS = list(TARGETS)
DATASET_IDS = {name: index for index, name in enumerate(DATASETS)}
TASK_NAMES = ["MI", "Emotion", "CS", "ADHD", "Workload"]
TASK_DESCRIPTIONS = [
    "Motor imagery EEG decoding",
    "Emotional state recognition from EEG",
    "Covert speech decoding from EEG",
    "Mental disorder diagnosis from EEG",
    "Mental workload assessment from EEG",
]
TASK_DATASETS = {
    "MI": ["MI_OpenBMI", "MI_BCIC_IV2a", "MI_BCIC_Upperlimb", "MI_ShanghaiU", "MI_HighGamma", "MI_Cho2017", "MI_Shin2017A", "MI_PhysioNet"],
    "Emotion": ["EMO_FACED", "EMO_SEED_3_seg4", "EMO_SEED_4_seg4", "EMO_SEED_5_seg4", "EMO_SEED_7_seg4"],
    "CS": ["CS_BCIC_Speech"],
    "ADHD": ["ADHD_AliMotie"],
    "Workload": ["Workload"],
}
DATASET_TO_TASK = [0] * len(DATASETS)
for task_index, task_name in enumerate(TASK_NAMES):
    for dataset_name in TASK_DATASETS[task_name]:
        DATASET_TO_TASK[DATASET_IDS[dataset_name]] = task_index


def instruction_text(dataset, mode):
    if mode == "generic":
        return INSTRUCTIONS[dataset][0]
    if mode == "specific":
        return INSTRUCTIONS[dataset][-1]
    if mode == "default":
        return INSTRUCTIONS["Default"][-1]
    raise ValueError(mode)


def load_text_assets(root):
    root = Path(root)
    labels = sorted({label for values in TARGETS.values() for label in values})
    label_indices = {label: index for index, label in enumerate(labels)}
    texts = set(labels)
    for values in INSTRUCTIONS.values():
        texts.update(values)
    texts.update(TASK_DESCRIPTIONS)
    embeddings = {text: np.load(root / f"{text}.npy").astype(np.float32) for text in texts}
    prototypes = np.asarray([embeddings[label] for label in labels], dtype=np.float32)
    return prototypes, label_indices, embeddings
