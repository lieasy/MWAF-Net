"""Single MWAF-Net training and evaluation; outputs are generated locally only."""

import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from .data import TASKS, load_prepared
from .loss import FocalLoss, mixup_data, mixup_criterion
from .model import MWAFNet

DEFAULT_TRAINING = dict(
    epochs=200,
    batch_size=16,
    learning_rate=0.001,
    weight_decay=0.000001,
    warmup_epochs=5,
    grad_accum_steps=2,
    grad_clip_norm=0.5,
    mixup_alpha=0.2,
    mixup_probability=0.8,
    patience=50,
    gamma=2.5,
    label_smoothing=0.2,
    train_augmentation=True,
    weighted_sampling=True,
    num_workers=0,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Segments(Dataset):
    def __init__(self, x, y, augmentation=False):
        self.x, self.y, self.augmentation = x, y, augmentation

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index].copy()
        if self.augmentation:
            if random.random() < 0.5:
                start = random.randint(0, 100)
                x = np.pad(x[start : start + 1300], (100, 100), mode="reflect")
            if random.random() < 0.3:
                x = x + np.random.normal(0, 0.05, x.shape)
            if random.random() < 0.4:
                x = x * random.uniform(0.8, 1.2)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(
            self.y[index], dtype=torch.long
        )


def read_config(path):
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("task") not in TASKS:
        raise ValueError("Configuration must specify one supported task")
    config["training"] = {**DEFAULT_TRAINING, **config.get("training", {})}
    return config


def get_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu")
    return device


def metrics(y, predicted, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y, predicted), 1)
    recall = np.divide(
        np.diag(cm), cm.sum(1), out=np.zeros(num_classes), where=cm.sum(1) > 0
    )
    return dict(
        accuracy_percent=float(np.trace(cm) / cm.sum() * 100),
        uar_percent=float(recall.mean() * 100),
        class_recall_percent=(recall * 100).tolist(),
        confusion_counts=cm.tolist(),
        samples=int(cm.sum()),
    )


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion=None):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite model logits")
        if criterion is not None:
            total_loss += float(criterion(logits, y)) * len(y)
        ys.extend(y.cpu().tolist())
        ps.extend(logits.argmax(1).cpu().tolist())
        samples += len(y)
    result = metrics(np.asarray(ys), np.asarray(ps), model.num_classes)
    if criterion is not None:
        result["loss"] = total_loss / samples
    return result


def train(config, data_dir, output_dir, seed=42, device="auto"):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a new/empty training output directory")
    t = config["training"]
    for key in ["epochs", "batch_size", "grad_accum_steps", "patience"]:
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (
        t["batch_size"] < 2
        or t["num_workers"] < 0
        or not 0 <= t["warmup_epochs"] < t["epochs"]
    ):
        raise ValueError("Use batch_size >= 2 and 0 <= warmup_epochs < epochs")
    seed_all(seed)
    device = get_device(device)
    arrays, meta = load_prepared(data_dir, config["task"])
    x, y = arrays["train"]
    if len(x) < 2:
        raise ValueError("BatchNorm training requires at least two samples")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if t["weighted_sampling"]:
        counts = np.bincount(y, minlength=len(meta["class_names"]))
        sampler = WeightedRandomSampler(
            torch.tensor(1.0 / counts[y], dtype=torch.double),
            len(y),
            replacement=True,
            generator=generator,
        )
    loader = DataLoader(
        Segments(x, y, t["train_augmentation"]),
        batch_size=t["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=t["num_workers"],
        generator=generator,
        drop_last=len(y) % t["batch_size"] == 1,
        pin_memory=device.type == "cuda",
    )
    val = DataLoader(
        Segments(*arrays["val"]),
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t["num_workers"],
    )
    model = MWAFNet(num_classes=len(meta["class_names"])).to(device)
    criterion = FocalLoss(
        model.num_classes, gamma=t["gamma"], label_smoothing=t["label_smoothing"]
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
    )
    warmup = t["warmup_epochs"]
    warm = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: min(1.0, (e + 1) / max(1, warmup))
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t["epochs"] - warmup, eta_min=1e-6
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    effective = dict(
        config=config,
        seed=seed,
        class_names=meta["class_names"],
        selection="validation UAR",
        initialization="archived global Conv1d initialization retained",
        singleton_remainder_dropped=len(y) % t["batch_size"] == 1,
        training_samples_per_epoch=len(loader.dataset)
        - int(len(y) % t["batch_size"] == 1),
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(effective, indent=2), encoding="utf-8"
    )
    best, stale = -float("inf"), 0
    for epoch in range(t["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if t["mixup_alpha"] > 0 and np.random.random() < t["mixup_probability"]:
                mixed, a, b, lam = mixup_data(batch_x, batch_y, t["mixup_alpha"])
                loss = mixup_criterion(criterion, model(mixed), a, b, lam)
            else:
                loss = criterion(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            accum = t["grad_accum_steps"]
            window_start = (i // accum) * accum
            window_size = min(accum, len(loader) - window_start)
            (loss / window_size).backward()
            total += float(loss.detach()) * len(batch_y)
            count += len(batch_y)
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        result = evaluate_loader(model, val, device, criterion)
        row = dict(
            epoch=epoch + 1,
            train_loss=total / count,
            validation=result,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        with (output_dir / "training_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch+1}/{t['epochs']} train_loss={total/count:.5f} val_UAR={result['uar_percent']:.2f}%",
            flush=True,
        )
        if result["uar_percent"] > best:
            best, stale = result["uar_percent"], 0
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    num_classes=model.num_classes,
                    task=meta["task"],
                    class_names=meta["class_names"],
                    config=config,
                    preprocessing={
                        k: meta[k]
                        for k in [
                            "target_sr",
                            "segment_seconds",
                            "min_seconds",
                            "global_normalization",
                        ]
                    },
                    epoch=epoch + 1,
                ),
                output_dir / "best.pt",
            )
        else:
            stale += 1
        if epoch < warmup:
            warm.step()
        else:
            cosine.step()
        if stale >= t["patience"]:
            break
    return output_dir / "best.pt"


def load_checkpoint(path, device="auto"):
    device = get_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint["task"] not in TASKS
        or checkpoint["class_names"] != TASKS[checkpoint["task"]]
    ):
        raise ValueError("Checkpoint task/class mapping is inconsistent")
    if checkpoint["num_classes"] != len(checkpoint["class_names"]):
        raise ValueError("Checkpoint output dimension is inconsistent")
    model = MWAFNet(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint, device
