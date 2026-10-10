"""Plot ECE408 CSR warp-per-row vs JDS thread-per-row benchmark results.

Usage:
    python3 plot_spmv.py spmv_results.csv

Creates spmv_kernel_time.png and spmv_speedup_vs_nnz_stddev.png.
Requires: matplotlib (pip install matplotlib).
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "spmv_results.csv")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV contains no results")

    names = [r["case"] for r in rows]
    csr = [float(r["csr_us"]) for r in rows]
    jds = [float(r["jds_us"]) for r in rows]
    stddev = [float(r["stddev_nnz_per_row"]) for r in rows]
    speedup = [float(r["csr_over_jds_speedup"]) for r in rows]

    positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([x - 0.2 for x in positions], csr, width=0.4,
           label="CSR warp-per-row")
    ax.bar([x + 0.2 for x in positions], jds, width=0.4,
           label="JDS thread-per-row")
    ax.set_xticks(positions, names)
    ax.set_ylabel("Median kernel time (us)")
    ax.set_title("SpMV: kernel time at constant total NNZ")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig("spmv_kernel_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stddev, speedup, marker="o")
    for x, y, name in zip(stddev, speedup, names):
        ax.annotate(name, (x, y), xytext=(5, 5),
                    textcoords="offset points")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Standard deviation of NNZ per row")
    ax.set_ylabel("CSR time / JDS time (higher = JDS faster)")
    ax.set_title("Impact of NNZ distribution on relative SpMV performance")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("spmv_speedup_vs_nnz_stddev.png", dpi=180)
    plt.close(fig)

    print("Saved spmv_kernel_time.png and spmv_speedup_vs_nnz_stddev.png")


if __name__ == "__main__":
    main()