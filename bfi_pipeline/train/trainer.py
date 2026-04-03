"""
train/trainer.py — Training loop for the BFI model.

Uses:
  • AdamW optimizer
  • CosineAnnealingLR (T_max = MAX_EPOCHS, eta_min = LR_MIN)
  • Early stopping on validation AUROC (patience = PATIENCE)
  • Gradient clipping (max_norm = GRAD_CLIP)
  • tqdm progress bars

Dataset format expected by the DataLoader:
  Each sample is a dict with keys: p1, p2, p3, gl, label (int64)
  Shapes: (L, D1), (L, D2), (L, D3), (L, D4), ()
"""

from __future__ import annotations

import copy
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from model.bfi_model import BFIModel
from model.losses    import BFILoss, compute_class_weights, derive_auxiliary_labels


# ─── Dataset ──────────────────────────────────────────────────────────────────

class BFIDataset(Dataset):
    """
    Flat dataset of (p1, p2, p3, gl, label) tensors.

    Parameters
    ----------
    samples : list of dicts, each with keys
              "p1" (L,D1), "p2" (L,D2), "p3" (L,D3), "gl" (L,D4),
              "label" (int), "patient_id" (str)
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "p1":    torch.tensor(s["p1"], dtype=torch.float32),
            "p2":    torch.tensor(s["p2"], dtype=torch.float32),
            "p3":    torch.tensor(s["p3"], dtype=torch.float32),
            "gl":    torch.tensor(s["gl"], dtype=torch.float32),
            "label": torch.tensor(s["label"], dtype=torch.long),
        }


def records_to_samples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten record → feature_seqs → samples list."""
    samples: List[Dict[str, Any]] = []
    for rec in records:
        pid    = rec["patient_id"]
        labels = rec.get("seq_labels", [])
        fseqs  = rec.get("feature_seqs", [])
        for i, fseq in enumerate(fseqs):
            if fseq is None:
                continue
            label = int(labels[i]) if i < len(labels) else 0
            samples.append({
                "p1":         fseq["p1"],
                "p2":         fseq["p2"],
                "p3":         fseq["p3"],
                "gl":         fseq["gl"],
                "label":      label,
                "patient_id": pid,
            })
    return samples


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """
    Trains a BFIModel on a fold's train/val split.

    Parameters
    ----------
    model      : BFIModel instance
    device     : torch device
    lr         : initial learning rate
    weight_decay : AdamW weight decay
    batch_size : mini-batch size
    max_epochs : maximum training epochs
    patience   : early-stopping patience (epochs without val AUROC improvement)
    grad_clip  : gradient clipping max norm
    aux_lambda : weight of auxiliary losses
    """

    def __init__(
        self,
        model:        BFIModel,
        device:       torch.device,
        lr:           float = cfg.LEARNING_RATE,
        weight_decay: float = cfg.WEIGHT_DECAY,
        batch_size:   int   = cfg.BATCH_SIZE,
        max_epochs:   int   = cfg.MAX_EPOCHS,
        patience:     int   = cfg.PATIENCE,
        grad_clip:    float = cfg.GRAD_CLIP,
        aux_lambda:   float = cfg.AUX_LAMBDA,
    ):
        self.model        = model.to(device)
        self.device       = device
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.grad_clip    = grad_clip

        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=cfg.LR_MIN
        )
        self._loss_fn: Optional[BFILoss] = None  # set after class weights known

    def _init_loss(self, train_samples: List[Dict[str, Any]]) -> None:
        labels = torch.tensor([s["label"] for s in train_samples])
        w      = compute_class_weights(labels)
        self._loss_fn = BFILoss(class_weights=w).to(self.device)

    def _make_loader(
        self, samples: List[Dict[str, Any]], shuffle: bool
    ) -> DataLoader:
        ds = BFIDataset(samples)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,       # avoid pickle issues with complex objects
            pin_memory=False,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        it = tqdm(loader, desc="  train", leave=False) if _HAS_TQDM else loader
        for batch in it:
            p1    = batch["p1"].to(self.device)
            p2    = batch["p2"].to(self.device)
            p3    = batch["p3"].to(self.device)
            gl    = batch["gl"].to(self.device)
            y     = batch["label"].to(self.device)

            # Forward
            class_probs, aux_scores, _ = self.model(p1, p2, p3, gl)

            # Derive auxiliary labels from feature values
            aux_labels = derive_auxiliary_labels(p1, p2, p3).to(self.device)

            loss, ce, aux = self._loss_fn(class_probs, y, aux_scores, aux_labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Returns (val_loss, val_auroc)."""
        self.model.eval()
        all_probs:  List[np.ndarray] = []
        all_labels: List[int]        = []
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            p1  = batch["p1"].to(self.device)
            p2  = batch["p2"].to(self.device)
            p3  = batch["p3"].to(self.device)
            gl  = batch["gl"].to(self.device)
            y   = batch["label"].to(self.device)

            class_probs, aux_scores, _ = self.model(p1, p2, p3, gl)
            aux_labels = derive_auxiliary_labels(p1, p2, p3).to(self.device)
            loss, _, _ = self._loss_fn(class_probs, y, aux_scores, aux_labels)

            total_loss += loss.item()
            n_batches  += 1

            all_probs.append(class_probs.cpu().numpy())
            all_labels.extend(y.cpu().tolist())

        probs_arr  = np.concatenate(all_probs, axis=0)   # (N, 3)
        labels_arr = np.array(all_labels)

        avg_loss = total_loss / max(n_batches, 1)

        # One-vs-rest macro AUROC
        try:
            if len(np.unique(labels_arr)) > 1:
                auroc = roc_auc_score(
                    labels_arr, probs_arr, multi_class="ovr", average="macro"
                )
            else:
                auroc = 0.5
        except Exception:
            auroc = 0.5

        return avg_loss, float(auroc)

    def fit(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples:   List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns
        -------
        dict with keys:
          "best_val_auroc", "best_epoch", "train_history", "val_history"
        """
        self._init_loss(train_samples)

        train_loader = self._make_loader(train_samples, shuffle=True)
        val_loader   = self._make_loader(val_samples,   shuffle=False)

        best_auroc     = -1.0
        best_state     = None
        best_epoch     = 0
        no_improve     = 0
        train_history: List[Dict] = []
        val_history:   List[Dict] = []

        for epoch in range(1, self.max_epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_auroc = self._evaluate(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_AUROC={val_auroc:.4f}  ({elapsed:.1f}s)"
            )

            train_history.append({"epoch": epoch, "loss": train_loss})
            val_history.append({"epoch": epoch, "loss": val_loss, "auroc": val_auroc})

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "best_val_auroc": best_auroc,
            "best_epoch":     best_epoch,
            "train_history":  train_history,
            "val_history":    val_history,
        }

    @torch.no_grad()
    def predict(
        self, samples: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a list of samples.

        Returns
        -------
        class_probs : (N, 3) float32
        bfi_scores  : (N,)   float32
        """
        self.model.eval()
        loader = self._make_loader(samples, shuffle=False)

        all_probs: List[np.ndarray] = []
        all_bfi:   List[np.ndarray] = []

        for batch in loader:
            p1 = batch["p1"].to(self.device)
            p2 = batch["p2"].to(self.device)
            p3 = batch["p3"].to(self.device)
            gl = batch["gl"].to(self.device)

            bfi, probs = self.model.predict_bfi(p1, p2, p3, gl)
            all_probs.append(probs.cpu().numpy())
            all_bfi.append(bfi.cpu().numpy())

        return (
            np.concatenate(all_probs, axis=0).astype(np.float32),
            np.concatenate(all_bfi,   axis=0).astype(np.float32),
        )
