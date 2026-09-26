"""Plotting utilities for A-BLAST validation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_efficiency(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.scatter(df["Average_total_time_s"], df["Average_MAE"], s=80)
    for _, row in df.iterrows():
        plt.annotate(row["Model"], (row["Average_total_time_s"], row["Average_MAE"]), xytext=(6, 6), textcoords="offset points")
    plt.xlabel("Average total computational time (s)")
    plt.ylabel("Average MAE")
    plt.title("Accuracy-efficiency positioning")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
