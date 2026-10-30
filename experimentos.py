"""
Projeto: CNNs são realmente invariantes?
Script principal dos experimentos (treino, avaliação, gráficos e tabelas).

Usa o script de apoio mnist_cnn_invariancia_augmentation.py para as
transformações de treino/teste e a CNN simples de referência.

O que este script faz:
1) separa 5.000 imagens do treino do MNIST como validação;
2) treina 4 arquiteturas (simples, profunda, com dropout, sem pooling),
   cada uma sem e com data augmentation (8 modelos no total);
3) avalia todos nos 6 conjuntos de teste (original, deslocado, rotacionado,
   com ruído, contraste alterado e combinado);
4) mede a acurácia em função da intensidade da transformação
   (deslocamento, rotação e ruído) e a consistência das predições;
5) mede a equivariância da primeira camada convolucional;
6) gera curvas de treino/validação, exemplos de acertos/erros, filtros e
   mapas de ativação na pasta resultados/.

Uso:
    python experimentos.py               # 5 épocas por modelo
    python experimentos.py --epochs 3    # execução mais rápida
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from mnist_cnn_invariancia_augmentation import (
    MNIST_MEAN,
    MNIST_STD,
    AddGaussianNoise,
    FixedRotate,
    FixedTranslate,
    SimpleCNN,
    make_test_transform,
    make_train_transform,
    seed_everything,
)


TEST_KINDS = ["original", "translated", "rotated", "noisy", "contrast", "combined"]
TEST_LABELS = {
    "original": "Original",
    "translated": "Deslocado",
    "rotated": "Rotacionado",
    "noisy": "Ruído",
    "contrast": "Contraste",
    "combined": "Combinado",
}
TRAIN_LABELS = {"plain": "sem aug.", "augmented": "com aug."}


# -----------------------------------------------------------------------------
# Arquiteturas
# -----------------------------------------------------------------------------
# Todas expõem `features` (parte convolucional) e `classifier`, e a primeira
# camada de `features` é sempre uma Conv2d, o que facilita as visualizações.

class DeepCNN(nn.Module):
    """Três blocos Conv -> ReLU -> MaxPool."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DropoutCNN(nn.Module):
    """Mesma CNN simples, com dropout antes da camada final."""

    def __init__(self, num_classes: int = 10, p: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class NoPoolCNN(nn.Module):
    """Mesmas convoluções da CNN simples, mas sem nenhum MaxPool."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODELS = {
    "simples": SimpleCNN,
    "profunda": DeepCNN,
    "dropout": DropoutCNN,
    "sem_pooling": NoPoolCNN,
}
MODEL_LABELS = {
    "simples": "CNN simples",
    "profunda": "CNN profunda",
    "dropout": "CNN + dropout",
    "sem_pooling": "CNN sem pooling",
}


# -----------------------------------------------------------------------------
# Dados
# -----------------------------------------------------------------------------

def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MNIST_MEAN[0]) / MNIST_STD[0]


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * MNIST_STD[0] + MNIST_MEAN[0]).clamp(0.0, 1.0)


def build_train_loaders(data_dir: str, val_size: int, batch_size: int, workers: int, seed: int,
                        train_limit: int | None = None):
    """Treino (sem/com augmentation) com os mesmos índices e validação sem augmentation."""
    plain = datasets.MNIST(data_dir, train=True, download=True, transform=make_train_transform(False))
    augmented = datasets.MNIST(data_dir, train=True, download=True, transform=make_train_transform(True))

    perm = torch.randperm(len(plain), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:val_size].tolist(), perm[val_size:].tolist()
    if train_limit:
        train_idx = train_idx[:train_limit]

    common = dict(num_workers=workers, persistent_workers=workers > 0)
    train_loaders = {
        "plain": DataLoader(Subset(plain, train_idx), batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed), **common),
        "augmented": DataLoader(Subset(augmented, train_idx), batch_size=batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(seed), **common),
    }
    val_loader = DataLoader(Subset(plain, val_idx), batch_size=1000, shuffle=False)
    return train_loaders, val_loader


def build_test_sets(data_dir: str, seed: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Materializa os 6 conjuntos de teste em tensores (o ruído fica fixo para todos os modelos)."""
    test_sets = {}
    for kind in TEST_KINDS:
        torch.manual_seed(seed)
        ds = datasets.MNIST(data_dir, train=False, download=True, transform=make_test_transform(kind))
        x = torch.stack([ds[i][0] for i in range(len(ds))])
        test_sets[kind] = (x, ds.targets.clone())
    return test_sets


def raw_test_images(data_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Imagens de teste em [0, 1], sem normalização (para as varreduras de intensidade)."""
    ds = datasets.MNIST(data_dir, train=False, download=True)
    return ds.data.float().div(255.0).unsqueeze(1), ds.targets.clone()


# -----------------------------------------------------------------------------
# Treino e avaliação
# -----------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch_size]).argmax(1) for i in range(0, len(x), batch_size)])


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total


def train_model(name: str, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float, seed: int) -> tuple[nn.Module, dict]:
    seed_everything(seed)
    model = MODELS[name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()

        val_loss, val_acc = evaluate_loader(model, val_loader, criterion)
        history["train_loss"].append(loss_sum / total)
        history["train_acc"].append(correct / total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  época {epoch:02d} | loss treino {loss_sum / total:.4f} | acc treino {correct / total:.4f}"
              f" | loss val {val_loss:.4f} | acc val {val_acc:.4f} | {time.time() - start:.0f}s", flush=True)

    return model, history


def sweep(model: nn.Module, raw_x: torch.Tensor, y: torch.Tensor, seed: int) -> dict:
    """Acurácia em função da intensidade de cada transformação."""
    out = {"shift": {}, "rotation": {}, "noise": {}}
    for s in range(0, 11):
        out["shift"][s] = (predict(model, normalize(FixedTranslate(dx=s, dy=0)(raw_x))) == y).float().mean().item()
    for a in range(0, 61, 5):
        out["rotation"][a] = (predict(model, normalize(FixedRotate(angle=a)(raw_x))) == y).float().mean().item()
    for std in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        torch.manual_seed(seed)
        out["noise"][std] = (predict(model, normalize(AddGaussianNoise(std=std)(raw_x))) == y).float().mean().item()
    return out


def consistency(model: nn.Module, test_sets: dict) -> dict:
    """Fração de imagens cuja predição não muda após a transformação (invariância empírica)."""
    base = predict(model, test_sets["original"][0])
    return {k: (predict(model, test_sets[k][0]) == base).float().mean().item() for k in TEST_KINDS if k != "original"}


@torch.no_grad()
def conv1_equivariance(model: nn.Module, raw_x: torch.Tensor, shift: int = 4) -> dict:
    """
    Compara conv1(T x) com T conv1(x), sendo T um deslocamento inteiro.
    Erro ~0 => equivariância. Também mede quanto a saída muda (invariância).
    """
    model.eval()
    x = normalize(raw_x[:500])
    # Desloca com preenchimento igual ao fundo normalizado (pixel 0 da imagem original)
    bg = normalize(torch.zeros(1)).item()
    tx = torch.full_like(x, bg)
    tx[..., :, shift:] = x[..., :, :-shift]

    conv1 = model.features[0]
    f_x, f_tx = conv1(x), conv1(tx)
    shifted_f_x = f_tx.clone()
    shifted_f_x[..., :, shift:] = f_x[..., :, :-shift]
    # Ignora as colunas da borda, onde o padding quebra a equivariância
    inner = slice(shift + 1, -1)
    eq_err = ((f_tx[..., inner] - shifted_f_x[..., inner]).norm() / f_tx[..., inner].norm()).item()
    inv_err = ((f_tx - f_x).norm() / f_x.norm()).item()
    return {"equivariance_rel_error": eq_err, "invariance_rel_change": inv_err}


# -----------------------------------------------------------------------------
# Figuras
# -----------------------------------------------------------------------------

def plot_curves(histories: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, len(MODELS), figsize=(4.2 * len(MODELS), 7), sharey="row")
    for j, name in enumerate(MODELS):
        for train_kind, color in [("plain", "tab:blue"), ("augmented", "tab:orange")]:
            h = histories[f"{name}__{train_kind}"]
            ep = range(1, len(h["train_loss"]) + 1)
            lbl = TRAIN_LABELS[train_kind]
            axes[0, j].plot(ep, h["train_loss"], color=color, ls="--", label=f"treino {lbl}")
            axes[0, j].plot(ep, h["val_loss"], color=color, label=f"validação {lbl}")
            axes[1, j].plot(ep, h["train_acc"], color=color, ls="--", label=f"treino {lbl}")
            axes[1, j].plot(ep, h["val_acc"], color=color, label=f"validação {lbl}")
        axes[0, j].set_title(MODEL_LABELS[name])
        axes[1, j].set_xlabel("época")
    axes[0, 0].set_ylabel("loss")
    axes[1, 0].set_ylabel("acurácia")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curvas_treino_validacao.png", dpi=130)
    plt.close(fig)


def plot_accuracy_bars(results: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    runs = [f"{m}__{t}" for m in MODELS for t in ("plain", "augmented")]
    width = 0.8 / len(runs)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, run in enumerate(runs):
        m, t = run.split("__")
        vals = [results[run][k] for k in TEST_KINDS]
        ax.bar(np.arange(len(TEST_KINDS)) + i * width - 0.4 + width / 2, vals, width,
               label=f"{MODEL_LABELS[m]} ({TRAIN_LABELS[t]})", color=colors[i * 2 + (t == "augmented")])
    ax.set_xticks(range(len(TEST_KINDS)))
    ax.set_xticklabels([TEST_LABELS[k] for k in TEST_KINDS])
    ax.set_ylabel("acurácia no teste")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "acuracia_por_transformacao.png", dpi=130)
    plt.close(fig)


def plot_sweeps(sweeps: dict, out: Path) -> None:
    titles = {"shift": ("Deslocamento horizontal", "pixels"),
              "rotation": ("Rotação", "graus"),
              "noise": ("Ruído gaussiano", "desvio padrão")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = dict(zip(MODELS, ["tab:blue", "tab:green", "tab:purple", "tab:red"]))
    for ax, key in zip(axes, titles):
        for run, s in sweeps.items():
            m, t = run.split("__")
            xs = [float(v) for v in s[key]]
            ax.plot(xs, list(s[key].values()), marker="o", ms=3, color=colors[m],
                    ls="-" if t == "augmented" else "--", label=f"{MODEL_LABELS[m]} ({TRAIN_LABELS[t]})")
        ax.set_title(titles[key][0])
        ax.set_xlabel(titles[key][1])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("acurácia")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "acuracia_vs_intensidade.png", dpi=130)
    plt.close(fig)


def plot_test_transforms(test_sets: dict, out: Path, n: int = 6) -> None:
    fig, axes = plt.subplots(len(TEST_KINDS), n, figsize=(1.6 * n, 1.7 * len(TEST_KINDS)))
    for i, kind in enumerate(TEST_KINDS):
        for j in range(n):
            axes[i, j].imshow(denormalize(test_sets[kind][0][j, 0]), cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
        axes[i, 0].set_title(TEST_LABELS[kind], fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "exemplos_transformacoes_teste.png", dpi=130)
    plt.close(fig)


def plot_examples(model: nn.Module, test_sets: dict, kinds: list[str], title: str, out_file: Path, n: int = 8) -> None:
    fig, axes = plt.subplots(2 * len(kinds), n, figsize=(1.5 * n, 3.4 * len(kinds)))
    for r, kind in enumerate(kinds):
        x, y = test_sets[kind]
        pred = predict(model, x)
        ok = torch.nonzero(pred == y).flatten()[:n]
        bad = torch.nonzero(pred != y).flatten()[:n]
        for row, idxs, tag in [(2 * r, ok, "acertos"), (2 * r + 1, bad, "erros")]:
            for j in range(n):
                ax = axes[row, j]
                ax.axis("off")
                if j < len(idxs):
                    k = idxs[j].item()
                    ax.imshow(denormalize(x[k, 0]), cmap="gray", vmin=0, vmax=1)
                    ax.set_title(f"real {y[k].item()} / pred {pred[k].item()}", fontsize=7,
                                 color="green" if tag == "acertos" else "red")
            axes[row, 0].text(-0.5, 0.5, f"{TEST_LABELS[kind]}\n{tag}", transform=axes[row, 0].transAxes,
                              ha="right", va="center", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_filters(models: dict, out: Path) -> None:
    runs = ["simples__plain", "simples__augmented"]
    fig, axes = plt.subplots(len(runs), 16, figsize=(16, 2.4))
    for i, run in enumerate(runs):
        w = models[run].features[0].weight.detach()[:, 0]
        for j in range(16):
            axes[i, j].imshow(w[j], cmap="bwr", vmin=-w.abs().max(), vmax=w.abs().max())
            axes[i, j].axis("off")
        m, t = run.split("__")
        axes[i, 0].set_title(f"{MODEL_LABELS[m]} ({TRAIN_LABELS[t]})", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "filtros_primeira_camada.png", dpi=130)
    plt.close(fig)


@torch.no_grad()
def plot_activations(model: nn.Module, raw_x: torch.Tensor, out: Path, idx: int = 0, n_maps: int = 8) -> None:
    """Mapas da 1ª camada para a imagem original e para a mesma imagem deslocada/rotacionada."""
    model.eval()
    img = raw_x[idx:idx + 1]
    variants = {
        "original": img,
        "deslocada (dx=5, dy=3)": FixedTranslate(dx=5, dy=3)(img),
        "rotacionada (20°)": FixedRotate(angle=20)(img),
    }
    conv1 = model.features[0]
    fig, axes = plt.subplots(len(variants), n_maps + 1, figsize=(1.6 * (n_maps + 1), 1.8 * len(variants)))
    for i, (name, v) in enumerate(variants.items()):
        maps = torch.relu(conv1(normalize(v)))[0]
        axes[i, 0].imshow(v[0, 0], cmap="gray")
        axes[i, 0].set_title(name, fontsize=8)
        for j in range(n_maps):
            axes[i, j + 1].imshow(maps[j], cmap="viridis")
            if i == 0:
                axes[i, j + 1].set_title(f"filtro {j}", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "mapas_ativacao.png", dpi=130)
    plt.close(fig)


def top_confusions(model: nn.Module, test_sets: dict, kind: str, k: int = 5) -> list:
    x, y = test_sets[kind]
    pred = predict(model, x)
    cm = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[t, p] += 1
    cm.fill_diagonal_(0)
    flat = [(cm[i, j].item(), i, j) for i in range(10) for j in range(10) if cm[i, j] > 0]
    return [{"real": i, "pred": j, "count": c} for c, i, j in sorted(flat, reverse=True)[:k]]


# -----------------------------------------------------------------------------
# Programa principal
# -----------------------------------------------------------------------------

def write_table(results: dict, out: Path) -> None:
    header = "| Modelo | Treino | " + " | ".join(TEST_LABELS[k] for k in TEST_KINDS) + " |"
    lines = [header, "|" + "---|" * (len(TEST_KINDS) + 2)]
    csv = ["modelo,treino," + ",".join(TEST_KINDS)]
    for m in MODELS:
        for t in ("plain", "augmented"):
            r = results[f"{m}__{t}"]
            lines.append(f"| {MODEL_LABELS[m]} | {TRAIN_LABELS[t]} | "
                         + " | ".join(f"{100 * r[k]:.2f}%" for k in TEST_KINDS) + " |")
            csv.append(f"{m},{t}," + ",".join(f"{r[k]:.4f}" for k in TEST_KINDS))
    (out / "tabela_resultados.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "resultados.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="./resultados")
    parser.add_argument("--sweep-size", type=int, default=2000,
                        help="nº de imagens de teste usadas nas varreduras de intensidade")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="limita o nº de imagens de treino (apenas para testes rápidos)")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "modelos").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    seed_everything(args.seed)
    train_loaders, val_loader = build_train_loaders(args.data_dir, 5000, args.batch_size, args.workers, args.seed,
                                                    args.train_limit)
    test_sets = build_test_sets(args.data_dir, args.seed)
    raw_x, raw_y = raw_test_images(args.data_dir)
    plot_test_transforms(test_sets, out)

    models, histories, results, sweeps, extras = {}, {}, {}, {}, {}
    for name in MODELS:
        for train_kind in ("plain", "augmented"):
            run = f"{name}__{train_kind}"
            n_params = sum(p.numel() for p in MODELS[name]().parameters())
            print(f"\n=== {MODEL_LABELS[name]} | {TRAIN_LABELS[train_kind]} | {n_params:,} parâmetros ===", flush=True)
            start = time.time()
            model, history = train_model(name, train_loaders[train_kind], val_loader, args.epochs, args.lr, args.seed)
            models[run], histories[run] = model, history
            torch.save(model.state_dict(), out / "modelos" / f"{run}.pt")

            results[run] = {k: (predict(model, x) == y).float().mean().item() for k, (x, y) in test_sets.items()}
            sweeps[run] = sweep(model, raw_x[:args.sweep_size], raw_y[:args.sweep_size], args.seed)
            extras[run] = {
                "params": n_params,
                "train_time_s": time.time() - start,
                "consistency": consistency(model, test_sets),
                "conv1": conv1_equivariance(model, raw_x),
                "top_confusions": {k: top_confusions(model, test_sets, k) for k in ("rotated", "combined")},
            }
            print("  teste: " + " | ".join(f"{TEST_LABELS[k]} {100 * v:.2f}%" for k, v in results[run].items()))

    write_table(results, out)
    for fname, obj in [("historico.json", histories), ("resultados.json", results),
                       ("varreduras.json", sweeps), ("extras.json", extras)]:
        (out / fname).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_curves(histories, out)
    plot_accuracy_bars(results, out)
    plot_sweeps(sweeps, out)
    plot_filters(models, out)
    plot_activations(models["simples__plain"], raw_x, out)
    plot_examples(models["simples__plain"], test_sets, ["translated", "rotated", "noisy", "combined"],
                  "CNN simples treinada SEM data augmentation", out / "exemplos_simples_sem_aug.png")
    plot_examples(models["simples__augmented"], test_sets, ["translated", "rotated", "noisy", "combined"],
                  "CNN simples treinada COM data augmentation", out / "exemplos_simples_com_aug.png")
    print(f"\nResultados salvos em {out.resolve()}")


if __name__ == "__main__":
    main()
