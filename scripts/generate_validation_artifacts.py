"""Generate validation tables and figures from saved manuscript results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

perf = pd.read_csv(RESULTS / "forecasting_performance.tsv", sep="\t")
time = pd.read_csv(RESULTS / "computational_efficiency.tsv", sep="\t")

# Accuracy-efficiency data
acc_eff = (
    perf.groupby("Model", as_index=False)
    .agg(Average_MSE=("MSE", "mean"), Average_MAE=("MAE", "mean"))
    .merge(time.groupby("Model", as_index=False).agg(Average_total_time_s=("Total_time_s", "mean")), on="Model")
)
acc_eff.to_csv(RESULTS / "accuracy_efficiency.tsv", sep="\t", index=False)

plt.figure(figsize=(6, 4.5))
plt.scatter(acc_eff["Average_total_time_s"], acc_eff["Average_MAE"], s=80)
for _, row in acc_eff.iterrows():
    plt.annotate(row["Model"], (row["Average_total_time_s"], row["Average_MAE"]), xytext=(6, 6), textcoords="offset points")
plt.xlabel("Average total computational time (s)")
plt.ylabel("Average MAE")
plt.title("Accuracy-efficiency positioning")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "accuracy_efficiency_plane.png", dpi=300)
plt.close()

# Paired Wilcoxon comparison
stat_rows = []
for metric in ["MSE", "MAE"]:
    for baseline in ["BLS", "Transformer"]:
        ablast_values = perf[perf["Model"] == "A-BLAST"].sort_values(["Dataset", "Horizon"])[metric].values
        base_values = perf[perf["Model"] == baseline].sort_values(["Dataset", "Horizon"])[metric].values
        stat, p_value = wilcoxon(ablast_values, base_values, zero_method="wilcox", alternative="two-sided", method="exact")
        stat_rows.append(
            {
                "Comparison": f"A-BLAST vs {baseline}",
                "Metric": metric,
                "Test": "Wilcoxon signed-rank",
                "p-value": round(float(p_value), 4),
                "Interpretation": "Significant" if p_value < 0.05 else "Not significant",
            }
        )
stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(RESULTS / "paired_statistical_comparison.tsv", sep="\t", index=False)

# Summary efficiency table
summary_eff = time.groupby("Model", as_index=False).agg(
    Average_training_time_s=("Training_time_s", "mean"),
    Average_prediction_time_s=("Prediction_time_s", "mean"),
    Average_total_time_s=("Total_time_s", "mean"),
)
summary_eff.to_csv(RESULTS / "computational_efficiency_summary.tsv", sep="\t", index=False)

# Comparative boxplots
merged = perf.merge(time, on=["Dataset", "Horizon", "Model"], how="left")
models = ["BLS", "Transformer", "A-BLAST"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, metric, title in zip(
    axes,
    ["MSE", "MAE", "Total_time_s"],
    ["(a) MSE", "(b) MAE", "(c) Computational time"],
):
    data = [merged[merged["Model"] == m][metric].values for m in models]
    ax.boxplot(data, tick_labels=models)
    ax.set_title(title)
    ax.set_ylabel(metric if metric != "Total_time_s" else "Total time (s)")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(FIGURES / "comparative_boxplots.png", dpi=300)
plt.close()

print("Generated validation artifacts:")
print(f"- {RESULTS / 'accuracy_efficiency.tsv'}")
print(f"- {RESULTS / 'paired_statistical_comparison.tsv'}")
print(f"- {RESULTS / 'computational_efficiency_summary.tsv'}")
print(f"- {FIGURES / 'accuracy_efficiency_plane.png'}")
print(f"- {FIGURES / 'comparative_boxplots.png'}")
