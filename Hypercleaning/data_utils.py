import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


# ── Corrupted MNIST / CIFAR-10 (TSP paper hypercleaning setup) ────────────────

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


def _random_subset(n_total, n_train, n_val, n_test, seed, name):
    if n_train + n_val + n_test > n_total:
        raise ValueError(f"n_train + n_val + n_test exceeds the {name} training set")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n_total, generator=generator)[:n_train + n_val + n_test]


def load_corrupted_mnist(n_train=5_000, n_val=5_000, n_test=10_000, corruption_frac=0.5,
                         seed=0, root="./data"):
    """Draw train/val/test splits from the MNIST training set and corrupt training labels.

    A fraction `corruption_frac` of the training labels is replaced by a uniformly random
    WRONG label (offset 1..9 mod 10). Validation and test labels are clean.

    Returns a dict with x_/y_ tensors for each split, `corrupted_mask` (bool, n_train)
    and `y_train_clean`.
    """
    full = datasets.MNIST(root=root, train=True, download=True)
    indices = _random_subset(len(full), n_train, n_val, n_test, seed, "MNIST")

    images = full.data[indices].float().unsqueeze(1) / 255.0      # (n, 1, 28, 28)
    labels = full.targets[indices].long()
    return _split_and_corrupt(images, labels, n_train, n_val, corruption_frac, seed)


def load_corrupted_cifar10(n_train=5_000, n_val=5_000, n_test=10_000, corruption_frac=0.5,
                           seed=0, root="./data"):
    """As load_corrupted_mnist, but for CIFAR-10 (images normalised per channel, (n, 3, 32, 32))."""
    full = datasets.CIFAR10(root=root, train=True, download=True)
    indices = _random_subset(len(full), n_train, n_val, n_test, seed, "CIFAR-10")

    images = torch.as_tensor(full.data[indices.numpy()]).permute(0, 3, 1, 2).float() / 255.0
    images = (images - torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)) / torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    labels = torch.as_tensor(full.targets)[indices].long()
    return _split_and_corrupt(images, labels, n_train, n_val, corruption_frac, seed)


def _split_and_corrupt(images, labels, n_train, n_val, corruption_frac, seed):
    x_train, y_train_clean = images[:n_train], labels[:n_train].clone()
    x_val, y_val = images[n_train:n_train + n_val], labels[n_train:n_train + n_val]
    x_test, y_test = images[n_train + n_val:], labels[n_train + n_val:]

    corruption_generator = torch.Generator().manual_seed(seed + 1)
    num_corrupted = int(corruption_frac * n_train)
    corrupted_indices = torch.randperm(n_train, generator=corruption_generator)[:num_corrupted]
    wrong_offsets = torch.randint(1, 10, (num_corrupted,), generator=corruption_generator)

    y_train = y_train_clean.clone()
    y_train[corrupted_indices] = (y_train[corrupted_indices] + wrong_offsets) % 10

    corrupted_mask = torch.zeros(n_train, dtype=torch.bool)
    corrupted_mask[corrupted_indices] = True

    return {
        "x_train": x_train, "y_train": y_train, "y_train_clean": y_train_clean,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "corrupted_mask": corrupted_mask,
    }


def make_datasets(splits):
    """Training samples carry their index so the upper-level model can look up its weight."""
    n_train = len(splits["y_train"])
    return {
        "train": TensorDataset(splits["x_train"], splits["y_train"], torch.arange(n_train)),
        "val":   TensorDataset(splits["x_val"], splits["y_val"]),
        "test":  TensorDataset(splits["x_test"], splits["y_test"]),
    }


# ── Training-loop helpers ─────────────────────────────────────────────────────

def cycling_iterator(loader: DataLoader):
    """Yields batches from loader indefinitely, restarting when exhausted."""
    while True:
        yield from loader


def move_batch(batch, device):
    return tuple(value.to(device) if torch.is_tensor(value) else value for value in batch)


def make_batch_iterators(train_dataset, val_dataset, batch_size=100, seed=0):
    """Shuffled, infinitely cycling train/val iterators with run-specific seeds."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed + 1000))
    return cycling_iterator(train_loader), cycling_iterator(val_loader)


def get_batch(train_iter, val_iter, device=torch.device("cpu")) -> dict:
    """Pull one batch from each iterator and return the data dict the loss functions expect:
    data['train'] = (images, labels, sample_ids),  data['val'] = (images, labels)."""
    return {
        "train": move_batch(next(train_iter), device),
        "val":   move_batch(next(val_iter), device),
    }
