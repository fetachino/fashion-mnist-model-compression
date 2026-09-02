#!/usr/bin/env python3
"""
CSCI 49000AIT - Homework 4 (Model Compression)
Name: Ahmed Balde
Student ID:

Instructions:
Fill in all sections labeled:  # TODO
Do not remove function names or change signatures.
Your code must run on CPU-only systems (no GPU required).
"""

import os
import copy
import random
import argparse
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# pruning utilities
import torch.nn.utils.prune as prune

# quantization
import torch.quantization as quant

# reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cpu") 

NUM_CLASSES = 10
CLASS_NAMES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# Default directories
DATA_DIR = "./data"
OUT_DIR = "./out_models"
os.makedirs(OUT_DIR, exist_ok=True)


# Data loaders
def get_dataloaders(data_dir: str = DATA_DIR,
                    batch_size: int = 128,
                    val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return train/val/test DataLoaders for FashionMNIST.
    """
    # transforms: simple normalization; keep as float32 single-channel
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # approx mean/std for FashionMNIST
    ])

    # Download datasets
    train_val = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    # split train into train/val
    n_total = len(train_val)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(train_val, [n_train, n_val],
                                              generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# Model Definition
class SimpleCNN(nn.Module):
    """
    A small CNN that is friendly to PyTorch module fusion (conv+relu pairs are named).
    Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> ReLU -> Dropout -> FC
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # features (name layers for fusion)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # output 28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # 14x14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  # 7x7

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

        # For quantization: a stub to indicate where quant/dequant occurs for static quant.
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        # quantize input for quantization-aware/static quant
        x = self.quant(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.dequant(x)
        return x

    def fuse_model(self):
        """
        Fuse modules to prepare for static quantization.
        Must match fused patterns; see PyTorch quantization docs.
        """
        # fuse conv+relu
        quant.fuse_modules(self, [['conv1', 'relu1'], ['conv2', 'relu2'], ['fc1', 'relu3']], inplace=True)


# Training & evaluation helpers
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    n_samples = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=1)
        running_loss += loss.item() * xb.size(0)
        running_correct += (preds == yb).sum().item()
        n_samples += xb.size(0)
    return running_loss / n_samples, running_correct / n_samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion=None) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    n_samples = 0
    all_preds = []
    all_targets = []
    for xb, yb in loader:
        out = model(xb)
        if criterion is not None:
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        running_correct += (preds == yb).sum().item()
        n_samples += xb.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    avg_loss = running_loss / n_samples if criterion is not None else 0.0
    avg_acc = running_correct / n_samples
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    return avg_loss, avg_acc, y_true, y_pred


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        optimizer, epochs: int = 10, criterion=None, lr_scheduler=None) -> Tuple[nn.Module, Dict]:
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if lr_scheduler:
            lr_scheduler.step()

        print(f"Epoch {ep:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, history


# Pruning
def apply_global_unstructured_pruning(model: nn.Module, amount: float):
    """
    Apply global unstructured L1 pruning to all Conv2d and Linear weights.
    amount: fraction of connections to prune (0.0 - 1.0)
    This function modifies model in-place and leaves pruning reparameterizations applied.
    """
    if amount <= 0.0:
        return

    params_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params_to_prune.append((module, "weight"))

    if not params_to_prune:
        return

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


def remove_pruning_reparametrization(model: nn.Module):
    """
    Remove the pruning reparametrization (so weights are permanently pruned).
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                # weight was not pruned; safe to ignore
                pass

def compute_sparsity(model: nn.Module) -> float:
    """
    Compute global sparsity fraction across Conv2d and Linear weights.
    Returns fraction of zeros among total parameters considered.
    """
    total_zeros = 0
    total_params = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total_params += w.numel()
            total_zeros += torch.sum(w == 0).item()

    return total_zeros / total_params if total_params > 0 else 0.0


def pruning_experiments(baseline_model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        test_loader: DataLoader,
                        criterion,
                        optimizer_fn,
                        prune_amounts: List[float],
                        fine_tune_epochs: int = 2) -> List[Dict]:
    """
    For each prune_amount (fraction), create a copy of baseline_model, prune weights globally,
    fine-tune for a small number of epochs, evaluate and record metrics.
    """
    results = []

    for amt in prune_amounts:
        print(f"\n=== Pruning experiment amount={amt} ===")

        # fresh copy of the baseline model
        model = copy.deepcopy(baseline_model).to(DEVICE)

        # apply global pruning
        apply_global_unstructured_pruning(model, amt)

        # fine-tune briefly if requested
        optimizer = optimizer_fn(model.parameters())
        if fine_tune_epochs > 0:
            model, _ = fit(model, train_loader, val_loader, optimizer,
                           epochs=fine_tune_epochs, criterion=criterion)

        # remove pruning reparam so weights are permanently pruned
        remove_pruning_reparametrization(model)

        # compute sparsity
        sparsity = compute_sparsity(model)

        # evaluate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)

        # save model
        fname = os.path.join(OUT_DIR, f"model_pruned_{int(amt * 100)}.pth")
        torch.save(model.state_dict(), fname)

        results.append({
            'prune_amount_requested': amt,
            'sparsity': sparsity,
            'val_acc': float(val_acc),
            'test_acc': float(test_acc),
            'model_state_path': fname,
            'y_true': y_true,
            'y_pred': y_pred
        })
        print(f"Pruning amt {amt} -> val_acc={val_acc:.4f} test_acc={test_acc:.4f} sparsity={sparsity:.4f}")

    return results



# Helper function to run calibration
def evaluate_for_calibration(model: nn.Module, loader: DataLoader, num_batches: int = None):
    """
    Run model in eval mode over the loader to calibrate observers (for static quantization).
    """
    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            model(xb)
            if num_batches is not None and i + 1 >= num_batches:
                break


# Post Training Static Quantization
def post_training_static_quantization(model: nn.Module, calib_loader: DataLoader, test_loader: DataLoader,
                                      criterion, backend: str = 'fbgemm') -> Dict:
    """
    Convert a CPU model to a statically quantized model (post-training quantization).
    If the requested quantized engine is not supported on this build of PyTorch,
    fall back to evaluating a float32 CPU copy so the script still runs.
    """
    # copy baseline model to CPU
    model_to_quant = copy.deepcopy(model).to("cpu")
    model_to_quant.eval()

    try:
        # Try to set backend and run true static quantization
        torch.backends.quantized.engine = backend

        # fuse modules
        model_to_quant.fuse_model()

        # set qconfig and prepare (insert observers)
        model_to_quant.qconfig = quant.get_default_qconfig(backend)
        quant.prepare(model_to_quant, inplace=True)

        # calibration on validation data
        evaluate_for_calibration(model_to_quant, calib_loader)

        # convert to quantized model
        quant.convert(model_to_quant, inplace=True)

        mode_desc = f"static quantized ({backend})"

    except RuntimeError as e:
        # Backend not supported on this build – fall back gracefully
        print(f"[WARNING] Quantized backend '{backend}' not supported on this PyTorch build:")
        print(f"          {e}")
        print("          Falling back to evaluating a float32 CPU model (no quantization).")

        # Just keep the float32 fused model (no prepare/convert)
        model_to_quant = copy.deepcopy(model).to("cpu")
        model_to_quant.eval()
        mode_desc = "float32 fallback (no quant)"

    # Evaluate (either quantized or fallback float) on test set
    test_loss, test_acc, y_true, y_pred = evaluate(model_to_quant, test_loader, criterion)
    print(f"{mode_desc} test_acc={test_acc:.4f} test_loss={test_loss:.4f}")

    fname = os.path.join(OUT_DIR, "model_static_quantized.pth")
    torch.save(model_to_quant.state_dict(), fname)

    return {
        'model': model_to_quant,
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'state_path': fname,
        'y_true': y_true,
        'y_pred': y_pred
    }

# Quantization Aware Training (QAT)
def quantization_aware_training(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                                test_loader: DataLoader, criterion, optimizer_fn,
                                qat_epochs: int = 5, backend: str = 'fbgemm') -> Dict:
    """
    Prepare model for QAT, fine-tune for a few epochs, then convert and evaluate.
    If quantized backend is not supported, fall back to float32 training on CPU.
    """
    # start from baseline on CPU
    model_qat = copy.deepcopy(model).to("cpu")

    try:
        # backend, fuse and set qconfig
        torch.backends.quantized.engine = backend
        model_qat.fuse_model()
        model_qat.qconfig = quant.get_default_qat_qconfig(backend)

        # insert fake-quant and observers
        quant.prepare_qat(model_qat, inplace=True)

        # train with QAT
        optimizer = optimizer_fn(model_qat.parameters())
        print("\nStarting QAT training...")
        model_qat, _ = fit(model_qat, train_loader, val_loader, optimizer,
                           epochs=qat_epochs, criterion=criterion)

        # convert to quantized model
        model_qat.eval()
        quant.convert(model_qat, inplace=True)

        mode_desc = f"QAT quantized ({backend})"

    except RuntimeError as e:
        print(f"[WARNING] QAT backend '{backend}' not supported on this PyTorch build:")
        print(f"          {e}")
        print("          Falling back to standard float32 fine-tuning on CPU (no QAT).")

        # Revert to plain float model on CPU and train normally
        model_qat = copy.deepcopy(model).to("cpu")
        optimizer = optimizer_fn(model_qat.parameters())
        model_qat, _ = fit(model_qat, train_loader, val_loader, optimizer,
                           epochs=qat_epochs, criterion=criterion)
        model_qat.eval()
        mode_desc = "float32 fallback (no QAT)"

    # evaluate
    test_loss, test_acc, y_true, y_pred = evaluate(model_qat, test_loader, criterion)
    print(f"{mode_desc} test_acc={test_acc:.4f}")

    fname = os.path.join(OUT_DIR, "model_qat_quantized.pth")
    torch.save(model_qat.state_dict(), fname)

    return {
        'model': model_qat,
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'state_path': fname,
        'y_true': y_true,
        'y_pred': y_pred
    }

# Summarize Experiments
def summarize_experiment_table(baseline_metrics: Dict,
                               pruning_results: List[Dict],
                               static_quant_res: Dict,
                               qat_res: Dict):
    rows = []
    # baseline
    rows.append({
        'method': 'baseline (float32)',
        'val_acc': baseline_metrics.get('val_acc', None),
        'test_acc': baseline_metrics.get('test_acc', None),
        'sparsity': 0.0,
        'notes': 'vanilla trained model'
    })
    # pruning rows
    for r in pruning_results:
        rows.append({
            'method': f'pruned {int(r["prune_amount_requested"]*100)}%',
            'val_acc': r['val_acc'],
            'test_acc': r['test_acc'],
            'sparsity': r['sparsity'],
            'notes': f'Model saved: {os.path.basename(r["model_state_path"])}'
        })
    # static quant
    rows.append({
        'method': 'post-training static quant',
        'val_acc': None,
        'test_acc': static_quant_res['test_acc'],
        'sparsity': None,
        'notes': f'state: {os.path.basename(static_quant_res["state_path"])}'
    })
    # QAT
    rows.append({
        'method': 'quantization-aware training (QAT)',
        'val_acc': None,
        'test_acc': qat_res['test_acc'],
        'sparsity': None,
        'notes': f'state: {os.path.basename(qat_res["state_path"])}'
    })

    df = pd.DataFrame(rows)
    print("\n=== Experiment Summary ===")
    print(df.to_string(index=False))
    summary_path = os.path.join(OUT_DIR, "experiment_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")
    return df


# Confusion Matrix Plotting
def plot_confusion(cm, classes=CLASS_NAMES, title="Confusion matrix"):
    """
    Plot confusion matrix using matplotlib.
    """
    import itertools
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)

    # threshold for text color
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(int(cm[i, j]), 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=7)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(fname, dpi=150)
    print(f"Saved confusion matrix to {fname}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="HW4 Compression Assignment - Solution script")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--baseline-epochs", type=int, default=8)   # keep small for demo
    parser.add_argument("--fine-tune-epochs", type=int, default=2)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--prune-amounts", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # 1) Baseline training
    baseline_model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    print("\n=== Training baseline model ===")
    baseline_model, history = fit(baseline_model, train_loader, val_loader, baseline_optimizer,
                                  epochs=args.baseline_epochs, criterion=criterion)

    # evaluate baseline on val/test
    val_loss, val_acc, _, _ = evaluate(baseline_model, val_loader, criterion)
    test_loss, test_acc, y_true_b, y_pred_b = evaluate(baseline_model, test_loader, criterion)
    print(f"\nBaseline Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    baseline_path = os.path.join(OUT_DIR, "model_baseline.pth")
    torch.save(baseline_model.state_dict(), baseline_path)

    baseline_metrics = {'val_acc': float(val_acc), 'test_acc': float(test_acc), 'path': baseline_path}

    # 2) Pruning experiments
    def opt_fn(params):
        return torch.optim.Adam(params, lr=1e-4)  # small fine-tune LR

    pruning_results = pruning_experiments(baseline_model, train_loader, val_loader, test_loader,
                                         criterion, opt_fn, prune_amounts=args.prune_amounts,
                                         fine_tune_epochs=args.fine_tune_epochs)

    # 3) Post-training static quantization (no retraining)
    # Use baseline model copy and calibrate on validation set
    print("\n=== Post-training static quantization (no retraining) ===")
    static_quant_res = post_training_static_quantization(baseline_model, val_loader, test_loader, criterion,
                                                         backend='fbgemm')

    # 4) Quantization-aware training (QAT)
    def opt_fn_qat(params):
        return torch.optim.SGD(params, lr=1e-4, momentum=0.9)

    print("\n=== Quantization-aware training (QAT) ===")
    qat_res = quantization_aware_training(baseline_model, train_loader, val_loader, test_loader,
                                         criterion, opt_fn_qat, qat_epochs=args.qat_epochs, backend='fbgemm')

    # 5) Summarize experiments and save
    df = summarize_experiment_table(baseline_metrics, pruning_results, static_quant_res, qat_res)

    # 6) Confusion matrix for the best model (choose highest test acc among baseline/pruned/static/qAT)
    candidates = [
        ('baseline', baseline_metrics['test_acc'], baseline_model, y_true_b, y_pred_b),
    ]
    for r in pruning_results:
        candidates.append((f"pruned_{int(r['prune_amount_requested']*100)}", r['test_acc'], None, r['y_true'], r['y_pred']))
    candidates.append(('static_quant', static_quant_res['test_acc'], static_quant_res['model'], static_quant_res['y_true'], static_quant_res['y_pred']))
    candidates.append(('qat_quant', qat_res['test_acc'], qat_res['model'], qat_res['y_true'], qat_res['y_pred']))

    best = max(candidates, key=lambda t: t[1])
    print(f"\nBest model by test acc: {best[0]} with acc={best[1]:.4f}")

    # If model object is not provided (pruned models saved to disk), we still have preds/targets
    _, best_acc, best_model_obj, best_y_true, best_y_pred = best
    cm = confusion_matrix(best_y_true, best_y_pred, labels=np.arange(NUM_CLASSES))
    plot_confusion(cm, title=f"Confusion matrix ({best[0]})")

    # Print classification report
    print("\nClassification Report (best):")
    print(classification_report(best_y_true, best_y_pred, target_names=CLASS_NAMES, zero_division=0))

    print("\nAll done. Models and summaries saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
