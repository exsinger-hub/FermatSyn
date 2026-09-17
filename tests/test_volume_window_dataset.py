import os

import numpy as np

from data.volume_window_dataset import SynthRADVolumeWindowDataset, parse_patient_slice


def _write_pair(path, left_value, right_value):
    left = np.full((256, 256), left_value, dtype=np.float32)
    right = np.full((256, 256), right_value, dtype=np.float32)
    np.save(path, np.concatenate([left, right], axis=1))


def test_parse_patient_slice():
    patient, index = parse_patient_slice("1BA005_123_concat.npy")
    assert patient == "1BA005"
    assert index == 123


def test_windows_never_cross_patient_or_slice_gap(tmp_path):
    train = tmp_path / "train"
    os.makedirs(train)
    for patient, indices in {"P001": [1, 2, 3, 5, 6], "P002": [7, 8, 9]}.items():
        for index in indices:
            _write_pair(train / f"{patient}_{index:03d}_concat.npy", index, -index)

    dataset = SynthRADVolumeWindowDataset(
        tmp_path, phase="train", window_size=3, stride=1, augment=False
    )
    assert len(dataset) == 2
    for item in dataset:
        indices = item["slice_indices"].tolist()
        assert indices[1] == indices[0] + 1
        assert indices[2] == indices[1] + 1
        assert item["A"].shape == (3, 1, 256, 256)
        assert item["B"].shape == (3, 1, 256, 256)


def test_stride_offset_rotates_between_epochs(tmp_path):
    train = tmp_path / "train"
    os.makedirs(train)
    for index in range(1, 9):
        _write_pair(train / f"P001_{index:03d}_concat.npy", index, -index)

    dataset = SynthRADVolumeWindowDataset(
        tmp_path, phase="train", window_size=3, stride=2, augment=False
    )
    epoch_zero = [item["center_slice_index"] for item in dataset]
    dataset.set_epoch(1)
    epoch_one = [item["center_slice_index"] for item in dataset]
    assert set(epoch_zero).isdisjoint(epoch_one)
    assert sorted(epoch_zero + epoch_one) == [2, 3, 4, 5, 6, 7]
