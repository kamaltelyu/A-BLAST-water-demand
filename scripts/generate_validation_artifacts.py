"""Compatibility helper for the final A-BLAST MethodsX repository.

Final manuscript artifacts are already archived under results/tables and figures.
For a full regeneration, use run_all.py or the one-click Colab notebook.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    ROOT / "results/tables/Table1_forecasting_performance.csv",
    ROOT / "results/tables/Table2_computational_efficiency.csv",
    ROOT / "results/tables/Table3_rolling_origin_robustness.csv",
    ROOT / "results/tables/Table4_ablation.csv",
    ROOT / "figures/Fig2_accuracy_efficiency.png",
    ROOT / "figures/Fig3_comparative_boxplots.png",
    ROOT / "figures/Fig4_forecast_visualization.png",
]
missing = [str(p.relative_to(ROOT)) for p in EXPECTED if not p.exists()]
if missing:
    raise SystemExit("Missing final artifacts:\n- " + "\n- ".join(missing))
print("Final manuscript artifacts are present.")
print("For a complete rerun: python run_all.py --out A-BLAST_MethodsX_RESULTS_v3_2")
