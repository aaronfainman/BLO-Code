from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import v2

# ── CIFAR-10 normalisation constants ─────────────────────────────────────────

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
CIFAR_NORMALIZE = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)

CIFAR_CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog",   "frog", "horse", "ship", "truck",
]

# ── Difficulty transforms ─────────────────────────────────────────────────────

DIFFICULTY_NAMES = ["easy", "medium", "hard"]
DIFFICULTY_NAMES = ["easy","hard"]

# _medium_transform = v2.Compose([
#     v2.GaussianNoise(mean=0.0, sigma=0.05),
# ])

# _hard_transform = v2.Compose([
#     v2.GaussianNoise(mean=0.0, sigma=0.2),
#     v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1)),
# ])

# _medium_transform = v2.Compose([
#     v2.GaussianNoise(mean=0.0, sigma=0.03),
#     v2.GaussianBlur(kernel_size=(1, 1), sigma=(0.2, 0.2)),
#     v2.RandomRotation(degrees=5, fill=0),
# ])

_hard_transform = v2.Compose([
    v2.GaussianNoise(mean=0.0, sigma=0.03),
    v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1)),
    v2.RandomRotation(degrees=15, fill=0),
    # v2.Resize(size=(30, 30), antialias=True),
    # v2.Pad(padding=1, fill=0),
])


TRANSFORM_PIPELINES = [
    nn.Identity(),
    # _medium_transform,
    _hard_transform,
]


def apply_difficulty_transform(image: torch.Tensor, difficulty: int) -> torch.Tensor:
    transformed = TRANSFORM_PIPELINES[difficulty](image)
    transformed = transformed.clamp(0.0, 1.0)
    return CIFAR_NORMALIZE(transformed)


# ── Dataset loading ───────────────────────────────────────────────────────────

CIFAR_ROOT = Path.cwd() / "cifar"
CIFAR_ROOT.mkdir(parents=True, exist_ok=True)

FULL_CIFAR_TRAIN = datasets.CIFAR10(
    root=CIFAR_ROOT,
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)


def make_seeded_difficulty_loaders(seed, n_train=6_000, n_val=1_000, batch_size=64):
    if n_train + n_val > len(FULL_CIFAR_TRAIN):
        raise ValueError("n_train + n_val exceeds the CIFAR-10 training set")

    split_generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(FULL_CIFAR_TRAIN), generator=split_generator)
    train_indices      = shuffled_indices[:n_train]
    validation_indices = shuffled_indices[n_train:n_train + n_val]

    difficulty_labels    = torch.arange(n_train) % len(DIFFICULTY_NAMES)
    difficulty_assignment = torch.randperm(n_train, generator=split_generator)
    difficulty_labels    = difficulty_labels[difficulty_assignment]

    transformed_images, class_labels = [], []
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for position, source_index in enumerate(train_indices):
            image, class_label = FULL_CIFAR_TRAIN[int(source_index)]
            difficulty = int(difficulty_labels[position])
            transformed_images.append(apply_difficulty_transform(image, difficulty))
            class_labels.append(class_label)

    training_dataset = TensorDataset(
        torch.stack(transformed_images),
        torch.tensor(class_labels, dtype=torch.long),
        difficulty_labels,
        train_indices,
    )

    val_difficulty_labels = torch.arange(n_val) % len(DIFFICULTY_NAMES)
    val_generator         = torch.Generator().manual_seed(seed + 1000)
    val_permutation       = torch.randperm(n_val, generator=val_generator)
    val_difficulty_labels = val_difficulty_labels[val_permutation]

    transformed_val_images, val_class_labels = [], []
    with torch.random.fork_rng():
        torch.manual_seed(seed + 1000)
        for position, source_index in enumerate(validation_indices):
            image, class_label = FULL_CIFAR_TRAIN[int(source_index)]
            difficulty = int(val_difficulty_labels[position])
            transformed_val_images.append(apply_difficulty_transform(image, difficulty))
            val_class_labels.append(class_label)

    validation_dataset = TensorDataset(
        torch.stack(transformed_val_images),
        torch.tensor(val_class_labels, dtype=torch.long),
    )

    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    return {
        "training_loader":          training_loader,
        "validation_loader":        validation_loader,
        "training_dataset":         training_dataset,
        "validation_dataset":       validation_dataset,
        "train_indices":            train_indices,
        "validation_indices":       validation_indices,
        "difficulty_labels":        difficulty_labels,
        "val_difficulty_labels":    val_difficulty_labels,
    }


def get_test_loader(seed=0, n=1_000, batch_size=64, root=CIFAR_ROOT):
    full_test = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
    )
    generator   = torch.Generator().manual_seed(seed)
    test_indices = torch.randperm(len(full_test), generator=generator)[:n]
    return DataLoader(Subset(full_test, test_indices.tolist()),
                      batch_size=batch_size, shuffle=False)


# ── Training-loop helpers ─────────────────────────────────────────────────────

def cycling_iterator(loader: DataLoader):
    """Yields batches from loader indefinitely, restarting when exhausted."""
    while True:
        yield from loader


def get_batch(train_iter, val_iter, test_iter) -> dict:
    """Pull one batch from each iterator and return a single data dict."""
    train_batch = next(train_iter)
    val_batch   = next(val_iter)
    test_batch  = next(test_iter)
    return {
        'features_train':   train_batch[0],
        'labels_train':     train_batch[1],
        'difficulty_train': train_batch[2],
        'indices_train':    train_batch[3],
        'features_val':     val_batch[0],
        'labels_val':       val_batch[1],
        'features_test':    test_batch[0],
        'labels_test':      test_batch[1],
    }
