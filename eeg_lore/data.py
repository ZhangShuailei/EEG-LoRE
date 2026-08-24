from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class DatasetSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    train_subjects: np.ndarray
    valid_x: np.ndarray
    valid_y: np.ndarray
    valid_subjects: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_subjects: np.ndarray


class SubjectStore:
    def __init__(self, root):
        self.root = Path(root)
        self.handles = {}

    def path(self, dataset):
        path = self.root / f"{dataset}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def handle(self, dataset):
        if dataset not in self.handles:
            self.handles[dataset] = h5py.File(self.path(dataset), "r", locking=False)
        return self.handles[dataset]

    def subjects(self, dataset):
        return list(self.handle(dataset).keys())

    def load_subjects(self, dataset, indices):
        handle = self.handle(dataset)
        names = self.subjects(dataset)
        xs, ys, subject_ids = [], [], []
        for index in indices:
            group = handle[names[index]]
            x = group["X"][:]
            y = group["Y"][:]
            xs.append(x)
            ys.append(y)
            subject_ids.append(np.full(len(x), index, dtype=np.int64))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(subject_ids)


def split_train_valid(x, y, subjects, ratio=0.2):
    step = int(1.0 / ratio)
    valid_indices = np.arange(0, len(x), step)
    train_indices = np.setdiff1d(np.arange(len(x)), valid_indices)
    return (
        x[train_indices],
        y[train_indices],
        subjects[train_indices],
        x[valid_indices],
        y[valid_indices],
        subjects[valid_indices],
    )


def split_subject_trials(x, y, ratio=0.75):
    boundary = int(round(ratio * len(x)))
    return (x[:boundary], y[:boundary]), (x[boundary:], y[boundary:])


def subject_array(handle, prefix, size):
    for suffix in ["S", "Sub", "Subj", "Subject", "Subjects", "s", "sub", "subj", "subject"]:
        key = f"{prefix}{suffix}"
        if key in handle:
            return handle[key][:]
    return np.full(size, -1, dtype=np.int64)


def load_presplit(path, dataset):
    with h5py.File(path, "r") as handle:
        source_train_x = handle["trainX"][:]
        source_train_y = handle["trainY"][:]
        source_valid_x = handle["validX"][:]
        source_valid_y = handle["validY"][:]
        source_test_x = handle["testX"][:]
        source_test_y = handle["testY"][:]
        source_train_subjects = subject_array(handle, "train", len(source_train_x))
        source_valid_subjects = subject_array(handle, "valid", len(source_valid_x))
        source_test_subjects = subject_array(handle, "test", len(source_test_x))
    if dataset == "ADHD_AliMotie":
        return DatasetSplit(
            source_train_x,
            source_train_y,
            source_train_subjects,
            source_valid_x,
            source_valid_y,
            source_valid_subjects,
            source_test_x,
            source_test_y,
            source_test_subjects,
        )
    train = split_train_valid(source_train_x, source_train_y, source_train_subjects, 0.2)
    if dataset == "EMO_SEED_4_seg4":
        test_x = np.concatenate((source_valid_x, source_test_x))
        test_y = np.concatenate((source_valid_y, source_test_y))
        test_subjects = np.concatenate((source_valid_subjects, source_test_subjects))
    else:
        test_x = source_valid_x
        test_y = source_valid_y
        test_subjects = source_valid_subjects
    return DatasetSplit(*train, test_x, test_y, test_subjects)


def load_dataset(dataset, root, valid_ratio=0.2):
    store = SubjectStore(root)
    if dataset in {"EMO_SEED_3_seg4", "EMO_SEED_4_seg4", "EMO_SEED_5_seg4", "EMO_SEED_7_seg4", "ADHD_AliMotie"}:
        return load_presplit(store.path(dataset), dataset)
    if dataset == "EMO_FACED":
        train_valid = store.load_subjects(dataset, range(100))
        test = store.load_subjects(dataset, range(100, 123))
        train = split_train_valid(*train_valid, valid_ratio)
        return DatasetSplit(*train, *test)
    subject_splits = {
        "MI_BCIC_IV2a": (range(0, 7), range(7, 9)),
        "MI_OpenBMI": (range(0, 42), range(42, 54)),
        "MI_BCIC_Upperlimb": (range(0, 11), range(11, 15)),
        "MI_ShanghaiU": (range(0, 20), range(20, 25)),
        "MI_HighGamma": (range(0, 10), range(10, 14)),
        "MI_Cho2017": (range(0, 40), range(40, 49)),
        "MI_Shin2017A": (range(0, 22), range(22, 28)),
        "MI_PhysioNet": (range(0, 80), range(80, 109)),
    }
    if dataset in subject_splits:
        train_indices, test_indices = subject_splits[dataset]
        train_valid = store.load_subjects(dataset, train_indices)
        test = store.load_subjects(dataset, test_indices)
        train = split_train_valid(*train_valid, valid_ratio)
        return DatasetSplit(*train, *test)
    if dataset == "CS_BCIC_Speech":
        train_x, train_y, train_subjects = [], [], []
        test_x, test_y, test_subjects = [], [], []
        for index in range(len(store.subjects(dataset))):
            x, y, _ = store.load_subjects(dataset, [index])
            if len(x) < 350:
                raise ValueError((dataset, index, len(x)))
            train_x.append(x[:250])
            train_y.append(y[:250])
            train_subjects.append(np.full(250, index, dtype=np.int64))
            test_x.append(x[300:350])
            test_y.append(y[300:350])
            test_subjects.append(np.full(50, index, dtype=np.int64))
        train_valid = (np.concatenate(train_x), np.concatenate(train_y), np.concatenate(train_subjects))
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y)
        test_subjects = np.concatenate(test_subjects)
        train = split_train_valid(*train_valid, valid_ratio)
        return DatasetSplit(*train, test_x, test_y, test_subjects)
    if dataset == "Workload":
        train_valid = store.load_subjects(dataset, range(32))
        test = store.load_subjects(dataset, range(32, 36))
        train = split_train_valid(*train_valid, valid_ratio)
        return DatasetSplit(*train, *test)
    raise ValueError(dataset)
