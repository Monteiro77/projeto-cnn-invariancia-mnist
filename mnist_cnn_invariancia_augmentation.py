"""
Projeto: CNNs são realmente invariantes?
Script de apoio para carregar MNIST e gerar datasets/dataloaders com transformações.

Este arquivo já cria:
1) treino sem data augmentation
2) treino com data augmentation
3) testes transformados: original, deslocado, rotacionado, com ruído,
   contraste alterado e transformação combinada.

Os alunos podem importar get_mnist_dataloaders() em outro notebook/script ou
executar este arquivo diretamente para verificar os tamanhos dos dataloaders.

Requisitos:
    pip install torch torchvision matplotlib
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


# -----------------------------------------------------------------------------
# Reprodutibilidade
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Define sementes para tornar os experimentos mais reprodutíveis."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Transformações customizadas para os conjuntos de teste
# -----------------------------------------------------------------------------

class FixedTranslate:
    """Desloca a imagem por um número fixo de pixels."""

    def __init__(self, dx: int = 4, dy: int = 4, fill: float = 0.0):
        self.dx = dx
        self.dy = dy
        self.fill = fill

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return TF.affine(
            x,
            angle=0.0,
            translate=[self.dx, self.dy],
            scale=1.0,
            shear=[0.0, 0.0],
            fill=self.fill,
        )


class FixedRotate:
    """Rotaciona a imagem por um ângulo fixo em graus."""

    def __init__(self, angle: float = 20.0, fill: float = 0.0):
        self.angle = angle
        self.fill = fill

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return TF.rotate(x, angle=self.angle, fill=self.fill)


class AddGaussianNoise:
    """Adiciona ruído gaussiano e mantém os pixels no intervalo [0, 1]."""

    def __init__(self, mean: float = 0.0, std: float = 0.18):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std + self.mean
        return torch.clamp(x + noise, 0.0, 1.0)


class FixedBrightnessContrast:
    """Altera brilho e contraste de forma fixa."""

    def __init__(self, brightness_factor: float = 0.65, contrast_factor: float = 1.8):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.adjust_brightness(x, self.brightness_factor)
        x = TF.adjust_contrast(x, self.contrast_factor)
        return torch.clamp(x, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Configuração dos dataloaders
# -----------------------------------------------------------------------------

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


@dataclass
class LoaderConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    seed: int = 42
    pin_memory: bool = True


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(MNIST_MEAN, MNIST_STD)


def make_train_transform(use_augmentation: bool) -> transforms.Compose:
    """
    Cria transformação de treino.

    Sem augmentation:
        ToTensor -> Normalize

    Com augmentation:
        ToTensor -> RandomAffine -> RandomApply(ruído) -> Normalize

    Observação: as transformações aleatórias são aplicadas apenas durante o treino.
    """
    if not use_augmentation:
        return transforms.Compose([
            transforms.ToTensor(),
            _normalize(),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.15, 0.15),
            scale=(0.90, 1.10),
            shear=5,
            fill=0.0,
        ),
        transforms.RandomApply([AddGaussianNoise(std=0.10)], p=0.35),
        _normalize(),
    ])


def make_test_transform(kind: str) -> transforms.Compose:
    """
    Cria transformações determinísticas para o conjunto de teste.

    kind pode ser:
        original
        translated
        rotated
        noisy
        contrast
        combined
    """
    base = [transforms.ToTensor()]

    if kind == "original":
        extra = []
    elif kind == "translated":
        extra = [FixedTranslate(dx=5, dy=3)]
    elif kind == "rotated":
        extra = [FixedRotate(angle=20)]
    elif kind == "noisy":
        extra = [AddGaussianNoise(std=0.22)]
    elif kind == "contrast":
        extra = [FixedBrightnessContrast(brightness_factor=0.60, contrast_factor=1.9)]
    elif kind == "combined":
        extra = [
            FixedTranslate(dx=3, dy=3),
            FixedRotate(angle=15),
            AddGaussianNoise(std=0.12),
            FixedBrightnessContrast(brightness_factor=0.75, contrast_factor=1.5),
        ]
    else:
        raise ValueError(
            f"Transformação desconhecida: {kind}. Use: original, translated, rotated, noisy, contrast ou combined."
        )

    return transforms.Compose(base + extra + [_normalize()])


def get_mnist_dataloaders(config: LoaderConfig | None = None) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Retorna dois dicionários:

    train_loaders:
        'train_plain'     -> treino sem data augmentation
        'train_augmented' -> treino com data augmentation

    test_loaders:
        'test_original'
        'test_translated'
        'test_rotated'
        'test_noisy'
        'test_contrast'
        'test_combined'
    """
    if config is None:
        config = LoaderConfig()

    seed_everything(config.seed)

    train_plain_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=make_train_transform(use_augmentation=False),
    )

    train_augmented_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=make_train_transform(use_augmentation=True),
    )

    train_loaders = {
        "train_plain": DataLoader(
            train_plain_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
        "train_augmented": DataLoader(
            train_augmented_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
    }

    test_kinds = ["original", "translated", "rotated", "noisy", "contrast", "combined"]
    test_loaders = {}

    for kind in test_kinds:
        dataset = datasets.MNIST(
            root=config.data_dir,
            train=False,
            download=True,
            transform=make_test_transform(kind),
        )
        test_loaders[f"test_{kind}"] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    return train_loaders, test_loaders


# -----------------------------------------------------------------------------
# Modelo simples opcional, apenas para dar um ponto de partida aos alunos
# -----------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """CNN pequena para MNIST. Os alunos devem criar variações desta arquitetura."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Calcula acurácia em um dataloader."""
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Treina por uma época e retorna a loss média."""
    model.train()
    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total += y.numel()

    return total_loss / total


def demo_train_and_evaluate(epochs: int = 2, use_augmented_training: bool = False) -> None:
    """
    Demonstração curta. Para o projeto completo, os alunos devem aumentar épocas,
    testar arquiteturas diferentes e registrar as curvas de treino/validação.
    """
    config = LoaderConfig(batch_size=128, num_workers=2)
    train_loaders, test_loaders = get_mnist_dataloaders(config)

    train_key = "train_augmented" if use_augmented_training else "train_plain"
    train_loader = train_loaders[train_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc_original = evaluate(model, test_loaders["test_original"], device)
        print(f"Época {epoch:02d} | loss treino = {loss:.4f} | acc teste original = {acc_original:.4f}")

    print("\nAcurácia final por conjunto de teste:")
    for name, loader in test_loaders.items():
        acc = evaluate(model, loader, device)
        print(f"{name:16s}: {acc:.4f}")


if __name__ == "__main__":
    # Verificação rápida dos dataloaders
    train_loaders, test_loaders = get_mnist_dataloaders(LoaderConfig(batch_size=64, num_workers=0))

    print("Treinos disponíveis:")
    for name, loader in train_loaders.items():
        x, y = next(iter(loader))
        print(f"{name:16s} -> batch imagens: {tuple(x.shape)}, batch labels: {tuple(y.shape)}")

    print("\nTestes disponíveis:")
    for name, loader in test_loaders.items():
        x, y = next(iter(loader))
        print(f"{name:16s} -> batch imagens: {tuple(x.shape)}, batch labels: {tuple(y.shape)}")

    # Para rodar uma demonstração curta, descomente uma das linhas abaixo:
    # demo_train_and_evaluate(epochs=2, use_augmented_training=False)
    # demo_train_and_evaluate(epochs=2, use_augmented_training=True)
