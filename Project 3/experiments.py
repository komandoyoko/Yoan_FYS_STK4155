from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from config import cfg, PLOTS_DIR
from train import train
from evaluate import evaluate_model


def run_single_experiment(
    tag: str,
    model_type: str | None = None,
    hidden_size: int | None = None,
    seq_len: int | None = None,
) -> Dict[str, Any]:
    """
    Modify config (in-place), train a model, evaluate on test set,
    and return a small dict with results.
    """
    if model_type is not None:
        cfg.model.model_type = model_type
    if hidden_size is not None:
        cfg.model.hidden_size = hidden_size
    if seq_len is not None:
        cfg.data.seq_len = seq_len

    print(
        f"\n=== {tag} ===\n"
        f"model_type = {cfg.model.model_type}, "
        f"hidden_size = {cfg.model.hidden_size}, "
        f"seq_len = {cfg.data.seq_len}\n"
    )

    # Train (this overwrites checkpoints/best_model.pt)
    stats = train()

    # Evaluate using evaluate_model() (loads the just-trained checkpoint)
    results = evaluate_model()
    test_acc = results.get("test_accuracy", None)
    test_loss = stats["test_loss"]

    print(f"{tag}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}\n")

    return {
        "tag": tag,
        "model_type": cfg.model.model_type,
        "hidden_size": cfg.model.hidden_size,
        "seq_len": cfg.data.seq_len,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }


def plot_bar(
    x_labels: List[str],
    values: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(x_labels))

    plt.figure()
    plt.bar(x, values)
    plt.xticks(x, x_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=200)
    plt.close()


def main():
    all_results: List[Dict[str, Any]] = []

    # -----------------------------
    # 1) Architecture comparison
    # -----------------------------
    arch_types = ["rnn", "gru", "lstm"]
    arch_results = []
    for mt in arch_types:
        res = run_single_experiment(
            tag=f"arch_{mt}",
            model_type=mt,
            hidden_size=64,       # fixed
            seq_len=200,          # fixed
        )
        arch_results.append(res)
        all_results.append(res)

    plot_bar(
        x_labels=arch_types,
        values=[r["test_accuracy"] for r in arch_results],
        xlabel="Architecture",
        ylabel="Test accuracy",
        title="Architecture comparison (RNN / GRU / LSTM)",
        filename="arch_comparison_accuracy.png",
    )

    # -----------------------------
    # 2) Hidden size comparison
    # -----------------------------
    cfg.model.model_type = "lstm"  # fix architecture
    hidden_sizes = [32, 64, 128]
    hid_results = []
    for hs in hidden_sizes:
        res = run_single_experiment(
            tag=f"hidden_{hs}",
            hidden_size=hs,
            seq_len=200,
        )
        hid_results.append(res)
        all_results.append(res)

    plot_bar(
        x_labels=[str(h) for h in hidden_sizes],
        values=[r["test_accuracy"] for r in hid_results],
        xlabel="Hidden size",
        ylabel="Test accuracy",
        title="Effect of hidden size (LSTM)",
        filename="hidden_size_comparison_accuracy.png",
    )

    # -----------------------------
    # 3) Sequence length comparison
    # -----------------------------
    cfg.model.model_type = "lstm"
    cfg.model.hidden_size = 64
    seq_lens = [100, 200, 400]
    seq_results = []
    for sl in seq_lens:
        res = run_single_experiment(
            tag=f"seq_{sl}",
            seq_len=sl,
        )
        seq_results.append(res)
        all_results.append(res)

    plot_bar(
        x_labels=[str(s) for s in seq_lens],
        values=[r["test_accuracy"] for r in seq_results],
        xlabel="Sequence length (samples)",
        ylabel="Test accuracy",
        title="Effect of sequence length (LSTM)",
        filename="seq_len_comparison_accuracy.png",
    )

    # Optionally save raw results as npz for later
    results_path = PLOTS_DIR / "experiments_summary.npz"
    np.savez(results_path, results=all_results)
    print(f"\nSaved experiment summary to {results_path}")
    print("Plots saved to", PLOTS_DIR)


if __name__ == "__main__":
    main()
