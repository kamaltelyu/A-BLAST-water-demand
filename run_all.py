from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data import acquire_datasets, sha256_file
from src.pipeline import ExperimentConfig, environment_info, load_chennai, load_melbourne, run_ablation, run_dataset
from src.reporting import (
    combine_tables_docx,
    make_fig2_accuracy_efficiency,
    make_fig3_boxplots,
    make_fig4_forecast,
    make_figS1_originwise_mae,
    make_table1,
    make_table2,
    make_table3,
    make_table4,
    make_table5_from_template,
    make_table6_from_template,
)


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def quick_config():
    return ExperimentConfig(
        train_ratio=0.70, val_ratio=0.15, lookback=30, stl_period=7, stl_robust=True,
        horizons=(7,), seeds=(11,), n_test_origins=3, max_train_samples=120,
        ridge_grid=(1e-3,), feature_nodes=16, enhancement_nodes=8, n_heads=4,
        num_layers=1, ff_dim=32, dropout=0.0,
        transformer_d_model=16, transformer_heads=4, transformer_layers=1,
        transformer_ff_dim=32, transformer_dropout=0.0, transformer_lr=1e-3,
        transformer_batch_size=32, transformer_max_epochs=2, transformer_patience=1,
        inference_warmup=1, inference_repeats=2, ablation_horizon=7,
    )


def synth_series(n: int, start: str, base: float, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = base + 0.01 * t + 10 * np.sin(2*np.pi*t/7) + rng.normal(0, 1, n)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(y, index=idx)


def main():
    ap = argparse.ArgumentParser(description="A-BLAST MethodsX v3.2 fixed-model rolling-origin manuscript-ready rerun")
    ap.add_argument("--out", default="A-BLAST_MethodsX_RESULTS_v3_2")
    ap.add_argument("--quick", action="store_true", help="Synthetic smoke test only; never use for manuscript")
    ap.add_argument("--chennai", help="Optional local Chennai raw CSV; skips download for Chennai")
    ap.add_argument("--melbourne", help="Optional local Melbourne raw CSV; skips download for Melbourne")
    ap.add_argument("--fresh", action="store_true", help="Delete an existing output directory and start from zero. Default is safe resume.")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    if root.exists() and args.fresh:
        shutil.rmtree(root)
    elif root.exists():
        print(f"RESUME MODE: preserving existing checkpoints in {root}")
    dirs = {
        "raw": root / "raw_results",
        "tables": root / "tables",
        "figures": root / "figures",
        "repro": root / "reproducibility",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    data_cache = root.parent / "_ABLAST_data_cache"
    data_cache.mkdir(parents=True, exist_ok=True)
    setup_logging(dirs["repro"] / "run_log.txt")

    code_snap = dirs["repro"] / "code_snapshot"
    code_snap.mkdir(parents=True, exist_ok=True)
    base_src = Path(__file__).resolve().parent
    for fname in ["run_all.py", "requirements.txt", "PROTOCOL_LOCK.json", "DATA_SOURCES.md", "MANUSCRIPT_OUTPUT_SPEC.md", "README_FINAL_ROLLING_ORIGIN.md"]:
        src_file = base_src / fname
        if src_file.exists():
            shutil.copy2(src_file, code_snap / fname)
    if (base_src / "src").exists():
        shutil.copytree(base_src / "src", code_snap / "src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    logging.info("==== A-BLAST MethodsX v3.2 fixed-model rolling-origin run/resume session started ====")
    logging.info("Output root: %s", root)
    logging.info("Mode: %s", "FRESH" if args.fresh else "RESUME-SAFE")

    cfg = quick_config() if args.quick else ExperimentConfig()
    env = environment_info()
    logging.info("Protocol: %s", json.dumps(asdict(cfg), default=list))
    logging.info("Environment: %s", json.dumps(env))

    if args.quick:
        datasets = {
            "Chennai": synth_series(900, "2010-01-01", 1000.0, 1),
            "Melbourne": synth_series(950, "2010-01-01", 600.0, 2),
        }
        profiles = {}
        for name, y in datasets.items():
            profiles[name] = {
                "dataset": name,
                "source_label": "SYNTHETIC SMOKE TEST — NOT FOR MANUSCRIPT",
                "source_url": "synthetic://smoke-test",
                "raw_path": "synthetic",
                "sha256": "NOT_APPLICABLE_SYNTHETIC",
                "size_bytes": 0,
                "target_variable": "Synthetic target",
                "unit": "arbitrary units",
                "start": str(y.index[0].date()), "end": str(y.index[-1].date()), "n": len(y),
                "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std(ddof=1)),
                "missing_target_values": 0, "missing_calendar_days": 0, "sampling_frequency": "Daily",
                "stl_period": cfg.stl_period, "missing_handling": "None",
            }
    else:
        if args.chennai or args.melbourne:
            if not (args.chennai and args.melbourne):
                ap.error("Provide both --chennai and --melbourne, or neither to use automatic downloads.")
            ch_path = Path(args.chennai)
            mel_path = Path(args.melbourne)
            datasets = {"Chennai": load_chennai(str(ch_path)), "Melbourne": load_melbourne(str(mel_path))}
            from src.pipeline import validate_daily_index
            profiles = {}
            for name, y, pth, source, target, unit in [
                ("Chennai", datasets["Chennai"], ch_path, "User-supplied local file", "Total reservoir water level", "million cubic feet (mcft)"),
                ("Melbourne", datasets["Melbourne"], mel_path, "User-supplied local file", "Daily bulk water supply volume (supplied_ML)", "megalitres (ML)"),
            ]:
                check = validate_daily_index(y, name)
                if check["n_missing_calendar_days"]:
                    raise ValueError(f"{name}: missing calendar days detected")
                profiles[name] = {
                    "dataset": name, "source_label": source, "source_url": source,
                    "raw_path": str(pth), "sha256": sha256_file(pth), "size_bytes": pth.stat().st_size,
                    "target_variable": target, "unit": unit,
                    "start": check["start"], "end": check["end"], "n": len(y),
                    "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std(ddof=1)),
                    "missing_target_values": int(y.isna().sum()), "missing_calendar_days": 0,
                    "sampling_frequency": "Daily", "stl_period": cfg.stl_period,
                    "missing_handling": "None; protocol aborts if target values or calendar dates are missing",
                }
        else:
            logging.info("Downloading exact public datasets automatically...")
            datasets, profiles = acquire_datasets(data_cache, cfg.stl_period)
            manifest_src = data_cache / "download_manifest.json"
            if manifest_src.exists():
                shutil.copy2(manifest_src, dirs["repro"] / "download_manifest.json")

    with open(dirs["repro"] / "dataset_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    with open(dirs["repro"] / "environment.json", "w") as f:
        json.dump(env, f, indent=2)
    with open(dirs["repro"] / "experiment_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    try:
        import subprocess
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (dirs["repro"] / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as e:
        logging.warning("Could not capture pip freeze: %s", e)

    all_metrics, all_summary = [], []
    all_timing, all_timing_summary = [], []
    all_preds, all_ablation, all_ablation_summary = [], [], []
    selected_all = {}
    status_path = dirs["repro"] / "RUN_STATUS.json"

    def save_status(stage, dataset=None, note=None):
        payload = {
            "stage": stage, "dataset": dataset, "note": note,
            "output_root": str(root),
            "resume_safe": True,
            "protocol_version": "v3.2 fixed-model rolling-origin",
            "checkpoint_granularity": "dataset × horizon × model × seed; ablation variant × seed",
            "test_origins_per_horizon": cfg.n_test_origins,
        }
        tmp = status_path.with_name(status_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(status_path)

    save_status("initialized")

    for name in ["Chennai", "Melbourne"]:
        save_status("main_experiment", name)
        logging.info("Running %s main experiments...", name)
        ds_out = dirs["raw"] / name.lower()
        metrics, summary, timing, timing_summary = run_dataset(name, datasets[name], str(ds_out), cfg)
        all_metrics.append(metrics); all_summary.append(summary)
        all_timing.append(timing); all_timing_summary.append(timing_summary)
        preds = pd.read_csv(ds_out / "predictions_by_seed_origin.csv")
        all_preds.append(preds)
        with open(ds_out / "selected_hyperparameters.json") as f:
            selected = json.load(f)
        selected_all[name] = selected
        logging.info("Running %s rolling-origin architectural ablation...", name)
        save_status("ablation", name)
        ab, absum = run_ablation(name, datasets[name], str(ds_out), cfg, selected)
        save_status("dataset_complete", name)
        all_ablation.append(ab); all_ablation_summary.append(absum)

    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    timing = pd.concat(all_timing, ignore_index=True)
    timing_summary = pd.concat(all_timing_summary, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    ablation = pd.concat(all_ablation, ignore_index=True)
    ablation_summary = pd.concat(all_ablation_summary, ignore_index=True)

    metrics.to_csv(dirs["raw"] / "metrics_by_seed_origin.csv", index=False)
    summary.to_csv(dirs["raw"] / "metrics_summary.csv", index=False)
    timing.to_csv(dirs["raw"] / "timing_by_seed.csv", index=False)
    timing_summary.to_csv(dirs["raw"] / "timing_summary.csv", index=False)
    preds.to_csv(dirs["raw"] / "predictions_by_seed_origin.csv", index=False)
    ablation.to_csv(dirs["raw"] / "ablation_by_seed_origin.csv", index=False)
    ablation_summary.to_csv(dirs["raw"] / "ablation_summary.csv", index=False)
    with open(dirs["repro"] / "selected_hyperparameters.json", "w") as f:
        json.dump(selected_all, f, indent=2)

    save_status("generating_manuscript_outputs")
    logging.info("Generating manuscript tables...")
    table_paths = []
    table_paths.append(dirs["tables"] / "Table1_forecasting_performance.docx")
    make_table1(summary, dirs["tables"] / "Table1_forecasting_performance.csv", table_paths[-1])
    table_paths.append(dirs["tables"] / "Table2_computational_efficiency.docx")
    make_table2(timing_summary, dirs["tables"] / "Table2_computational_efficiency.csv", table_paths[-1])
    table_paths.append(dirs["tables"] / "Table3_rolling_origin_robustness.docx")
    make_table3(metrics, dirs["tables"] / "Table3_rolling_origin_robustness.csv", table_paths[-1])
    table_paths.append(dirs["tables"] / "Table4_ablation.docx")
    make_table4(ablation_summary, dirs["tables"] / "Table4_ablation.csv", table_paths[-1])

    base = Path(__file__).resolve().parent
    exp_template = base / "templates" / "Experimental_setup_framework.docx"
    ds_template = base / "templates" / "Dataset_profile_Table6.docx"
    table5 = dirs["tables"] / "Table5_experimental_setup.docx"
    make_table5_from_template(exp_template, table5, asdict(cfg), env, selected_all)
    table_paths.append(table5)
    table6 = dirs["tables"] / "Table6_dataset_profile.docx"
    make_table6_from_template(ds_template, table6, profiles)
    table_paths.append(table6)
    combine_tables_docx(table_paths, dirs["tables"] / "Tables1-6_manuscript_ready.docx")

    logging.info("Generating manuscript figures...")
    make_fig2_accuracy_efficiency(metrics, timing, dirs["figures"] / "Fig2_accuracy_efficiency.png", dirs["figures"] / "Fig2_accuracy_efficiency.pdf")
    make_fig3_boxplots(metrics, timing, dirs["figures"] / "Fig3_comparative_boxplots.png", dirs["figures"] / "Fig3_comparative_boxplots.pdf")
    plot_h = 96 if 96 in cfg.horizons else cfg.horizons[0]
    central_origin = f"O{(cfg.n_test_origins + 1)//2}"
    make_fig4_forecast(preds, dirs["figures"] / "Fig4_forecast_visualization.png", dirs["figures"] / "Fig4_forecast_visualization.pdf", horizon=plot_h, origin_id=central_origin)
    make_figS1_originwise_mae(metrics, dirs["figures"] / "FigS1_rolling_origin_relative_MAE.png", dirs["figures"] / "FigS1_rolling_origin_relative_MAE.pdf")

    checklist = root / "MANUSCRIPT_OUTPUT_CHECKLIST.md"
    checklist.write_text(
        "# A-BLAST MethodsX v3.2 final rolling-origin outputs\n\n"
        "Generated automatically from one locked, resume-safe run.\n\n"
        "## Test design\n"
        f"- {cfg.n_test_origins} evenly spaced feasible rolling forecast origins per dataset/horizon.\n"
        "- Final seed-level model fitted once on train+validation and held fixed across test origins.\n"
        "- Earlier test observations may be used only as past observed inputs at later origins; no test-period refitting.\n"
        "- STL is recomputed from history strictly before each forecast origin.\n\n"
        "## Tables\n"
        "- Table 1: rolling-origin forecasting performance; seed-averaged per origin, then mean ± SD across origins\n"
        "- Table 2: model training/inference efficiency; common STL preprocessing excluded from model timing\n"
        "- Table 3: unit-free paired percentage robustness comparison across dataset × horizon × origin × seed\n"
        "- Table 4: true architectural ablation at the locked horizon across rolling origins\n"
        "- Table 5: experimental setup populated from locked config/runtime environment\n"
        "- Table 6: dataset profile populated from exact downloaded files, including SHA-256\n\n"
        "## Figures\n"
        "- Fig. 2: dataset-specific accuracy–efficiency panels (no cross-unit MAE aggregation)\n"
        "- Fig. 3: dataset-specific 2×3 error/time boxplots (no cross-unit error pooling)\n"
        f"- Fig. 4: 2×3 forecast visualization at H={plot_h}, pre-specified central origin {central_origin}, mean ±1 SD across seeds\n"
        "- Fig. S1: relative MAE across all rolling origins, normalized to Transformer within each dataset/horizon/origin\n\n"
        "Graphical Abstract and Fig. 1 are static methodological diagrams and are intentionally not regenerated from experimental results.\n",
        encoding="utf-8",
    )

    save_status("archiving")
    archive = shutil.make_archive(str(root), "zip", root_dir=root)
    save_status("complete", note=f"Final ZIP: {archive}")
    logging.info("Completed. Results archive: %s", archive)
    print(f"\nFINAL_RESULTS_DIR={root}")
    print(f"FINAL_RESULTS_ZIP={archive}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)
