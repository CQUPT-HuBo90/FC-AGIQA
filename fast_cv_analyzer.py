#!/usr/bin/env python3
"""
当前5折交叉验证结果分析器。

支持三类来源:
1. ckpts/best_model_<dataset>_fold*.pth 的checkpoint元数据
2. ablation_results.csv 的run_id + dataset + ablation_mode聚合
3. cross_dataset_results.csv 的source_dataset + target_dataset聚合
"""

import argparse
import os
from glob import glob

import pandas as pd

try:
    import torch
except ImportError:
    torch = None


EXPECTED_FOLDS = 5
CV_PROTOCOL = "5-fold-cv_folds"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_label_norm(value):
    if isinstance(value, dict):
        return value.get("type", "none")
    if pd.isna(value):
        return "none"
    return str(value)


def load_checkpoint_metadata_only(checkpoint_path):
    """只加载模型检查点的元数据，不把权重保留在内存里。"""
    if torch is None:
        print("当前环境未安装torch，无法读取checkpoint；请改用 --results_csv 分析CSV。")
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        filename = os.path.basename(checkpoint_path)
        dataset_from_name = filename.replace("best_model_", "").split("_fold")[0]

        label_norm = checkpoint.get("label_norm", "none")
        metadata = {
            "path": checkpoint_path,
            "dataset": checkpoint.get("benchmark", dataset_from_name),
            "ablation_mode": checkpoint.get("ablation_mode", "full"),
            "run_id": checkpoint.get("run_id", ""),
            "epoch": checkpoint.get("epoch", "Unknown"),
            "srcc": checkpoint.get("rho_s", 0.0),
            "plcc": checkpoint.get("rho_p", 0.0),
            "krcc": checkpoint.get("rho_k", 0.0),
            "fold": checkpoint.get("fold", "Unknown"),
            "score_type": checkpoint.get("score_type", "Unknown"),
            "cv_protocol": checkpoint.get("cv_protocol", CV_PROTOCOL),
            "quality_preset": checkpoint.get("quality_preset", "none"),
            "train_aug_policy": checkpoint.get("train_aug_policy", "legacy"),
            "scheduler": checkpoint.get("scheduler", ""),
            "use_ema": checkpoint.get("use_ema", False),
            "rank_loss_type": checkpoint.get("rank_loss_type", "hard"),
            "label_norm": _normalize_label_norm(label_norm),
            "label_norm_mean": checkpoint.get("label_norm_mean", ""),
            "label_norm_std": checkpoint.get("label_norm_std", ""),
            "freeze_text_encoder": checkpoint.get("freeze_text_encoder", False),
            "train_srcc": checkpoint.get("train_rho_s", 0.0),
            "train_plcc": checkpoint.get("train_rho_p", 0.0),
            "train_krcc": checkpoint.get("train_rho_k", 0.0),
            "generalization_gap_srcc": checkpoint.get("generalization_gap_srcc", 0.0),
            "use_balanced_sampling": checkpoint.get("use_balanced_sampling", False),
            "num_quality_bins": checkpoint.get("num_quality_bins", ""),
        }
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metadata
    except Exception as exc:
        print(f"无法加载 {os.path.basename(checkpoint_path)}: {exc}")
        return None


def detect_available_datasets(ckpts_dir="ckpts"):
    datasets = set()
    for model_file in glob(os.path.join(ckpts_dir, "best_model_*_fold*.pth")):
        filename = os.path.basename(model_file)
        datasets.add(filename.replace("best_model_", "").split("_fold")[0])
    return sorted(datasets)


def summarize_dataframe(df, dataset_name=None, expected_folds=EXPECTED_FOLDS):
    if df.empty:
        return None

    df = df.copy()
    df["fold"] = df["fold"].apply(_safe_int)
    for col in ["srcc", "plcc", "krcc"]:
        df[col] = df[col].apply(_safe_float)
    df = df.sort_values("fold")

    folds = sorted(df["fold"].dropna().unique().tolist())
    missing_folds = [fold for fold in range(1, expected_folds + 1) if fold not in folds]
    status = "完整" if not missing_folds and len(folds) == expected_folds else f"缺失fold: {missing_folds}"

    best_idx = df["srcc"].idxmax()
    best = df.loc[best_idx]

    return {
        "dataset": dataset_name or df["dataset"].iloc[0],
        "ablation_mode": df["ablation_mode"].iloc[0] if "ablation_mode" in df else "full",
        "run_id": df["run_id"].iloc[0] if "run_id" in df else "",
        "num_folds": len(folds),
        "expected_folds": expected_folds,
        "folds": folds,
        "missing_folds": missing_folds,
        "status": status,
        "srcc_mean": df["srcc"].mean(),
        "srcc_std": df["srcc"].std(ddof=1) if len(df) > 1 else 0.0,
        "plcc_mean": df["plcc"].mean(),
        "plcc_std": df["plcc"].std(ddof=1) if len(df) > 1 else 0.0,
        "krcc_mean": df["krcc"].mean(),
        "krcc_std": df["krcc"].std(ddof=1) if len(df) > 1 else 0.0,
        "best_fold": best["fold"],
        "best_srcc": best["srcc"],
        "results_df": df,
    }


def print_summary(result, show_details=True):
    if result is None:
        return

    title = f"{result['dataset']} | {CV_PROTOCOL}"
    if result.get("run_id"):
        title += f" | run_id={result['run_id']}"
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Fold状态: {result['num_folds']}/{result['expected_folds']} ({result['status']})")
    print(f"SRCC: {result['srcc_mean']:.4f} ± {result['srcc_std']:.4f}")
    print(f"PLCC: {result['plcc_mean']:.4f} ± {result['plcc_std']:.4f}")
    print(f"KRCC: {result['krcc_mean']:.4f} ± {result['krcc_std']:.4f}")
    print(f"最佳Fold: {result['best_fold']} (SRCC={result['best_srcc']:.4f})")
    if "generalization_gap_srcc" in result["results_df"].columns:
        gaps = result["results_df"]["generalization_gap_srcc"].apply(_safe_float)
        print(f"泛化Gap(train-val SRCC): {gaps.mean():.4f}")

    if not show_details:
        return

    df = result["results_df"].sort_values("fold")
    print("-" * 80)
    print(f"{'Fold':>4} {'Epoch':>6} {'SRCC':>8} {'PLCC':>8} {'KRCC':>8} {'TrainS':>8} {'Gap':>8} {'Run ID':>22}")
    print("-" * 80)
    for _, row in df.iterrows():
        run_id = str(row.get("run_id", ""))[-22:]
        print(
            f"{_safe_int(row['fold']):>4} {str(row.get('epoch', '')):>6} "
            f"{_safe_float(row['srcc']):>8.4f} {_safe_float(row['plcc']):>8.4f} "
            f"{_safe_float(row['krcc']):>8.4f} {_safe_float(row.get('train_srcc', 0.0)):>8.4f} "
            f"{_safe_float(row.get('generalization_gap_srcc', 0.0)):>8.4f} {run_id:>22}"
        )


def quick_analyze_cv_results(dataset_name, ckpts_dir="ckpts", expected_folds=EXPECTED_FOLDS):
    pattern = os.path.join(ckpts_dir, f"best_model_{dataset_name}_fold*.pth")
    model_files = sorted(glob(pattern))
    if not model_files:
        print(f"未找到数据集 {dataset_name} 的checkpoint: {pattern}")
        return None

    rows = []
    for model_file in model_files:
        metadata = load_checkpoint_metadata_only(model_file)
        if metadata:
            rows.append(metadata)

    result = summarize_dataframe(pd.DataFrame(rows), dataset_name, expected_folds)
    print_summary(result)
    return result


def quick_analyze_with_csv_fallback(dataset_name, ckpts_dir="ckpts", expected_folds=EXPECTED_FOLDS, csv_path="ablation_results.csv"):
    result = quick_analyze_cv_results(dataset_name, ckpts_dir, expected_folds)
    if result is not None:
        return result
    if os.path.exists(csv_path):
        print(f"回退到结果CSV分析: {csv_path}")
        return analyze_results_csv(csv_path, dataset=dataset_name, expected_folds=expected_folds)
    return None


def analyze_all_datasets(ckpts_dir="ckpts", expected_folds=EXPECTED_FOLDS):
    results = []
    for dataset in detect_available_datasets(ckpts_dir):
        result = quick_analyze_cv_results(dataset, ckpts_dir, expected_folds)
        if result:
            results.append(result)

    if results:
        print("\n" + "=" * 80)
        print(f"全数据集汇总 | {CV_PROTOCOL}")
        print("=" * 80)
        print(f"{'数据集':>15} {'Folds':>7} {'SRCC':>14} {'PLCC':>14} {'KRCC':>14} {'状态':>14}")
        for result in results:
            print(
                f"{result['dataset']:>15} {result['num_folds']:>2}/{result['expected_folds']:<4} "
                f"{result['srcc_mean']:.4f}±{result['srcc_std']:.4f} "
                f"{result['plcc_mean']:.4f}±{result['plcc_std']:.4f} "
                f"{result['krcc_mean']:.4f}±{result['krcc_std']:.4f} "
                f"{result['status']:>14}"
            )

    return results


def analyze_results_csv(csv_path, dataset=None, run_id=None, ablation_mode=None, expected_folds=EXPECTED_FOLDS):
    if not os.path.exists(csv_path):
        print(f"结果CSV不存在: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"结果CSV为空: {csv_path}")
        return []

    df = df.rename(columns={"SRCC": "srcc", "PLCC": "plcc", "KRCC": "krcc", "best_epoch": "epoch"})
    if {"source_dataset", "target_dataset"}.issubset(df.columns):
        return analyze_cross_dataset_csv_dataframe(df, dataset=dataset, run_id=run_id)

    for required in ["dataset", "ablation_mode", "fold", "srcc", "plcc", "krcc", "run_id"]:
        if required not in df.columns:
            if required == "run_id":
                df[required] = ""
            elif required == "ablation_mode":
                df[required] = "full"
            else:
                raise ValueError(f"CSV缺少必要列: {required}")

    for optional, default in [
        ("quality_preset", "none"),
        ("rank_loss_type", "hard"),
        ("label_norm", "none"),
        ("label_norm_mean", ""),
        ("label_norm_std", ""),
        ("freeze_text_encoder", False),
        ("train_SRCC", 0.0),
        ("train_PLCC", 0.0),
        ("train_KRCC", 0.0),
        ("generalization_gap_srcc", 0.0),
        ("use_balanced_sampling", False),
        ("num_quality_bins", ""),
    ]:
        if optional not in df.columns:
            df[optional] = default
    df = df.rename(columns={"train_SRCC": "train_srcc", "train_PLCC": "train_plcc", "train_KRCC": "train_krcc"})
    df["label_norm"] = df["label_norm"].apply(_normalize_label_norm)

    if dataset:
        df = df[df["dataset"] == dataset]
    if run_id:
        df = df[df["run_id"] == run_id]
    if ablation_mode:
        df = df[df["ablation_mode"] == ablation_mode]

    results = []
    group_cols = ["run_id", "dataset", "ablation_mode"]
    for (group_run_id, group_dataset, group_ablation), group in df.groupby(group_cols, dropna=False):
        group = group.sort_values("epoch").drop_duplicates(subset=["fold"], keep="last")
        result = summarize_dataframe(group, group_dataset, expected_folds)
        if result:
            result["run_id"] = group_run_id
            result["ablation_mode"] = group_ablation
            result["rank_loss_type"] = str(group["rank_loss_type"].iloc[0])
            result["label_norm"] = str(group["label_norm"].iloc[0])
            result["freeze_text_encoder"] = str(group["freeze_text_encoder"].iloc[0])
            result["quality_preset"] = str(group["quality_preset"].iloc[0])
            result["use_balanced_sampling"] = str(group["use_balanced_sampling"].iloc[0])
            print_summary(result, show_details=True)
            results.append(result)

    if results:
        print("\n" + "=" * 80)
        print(f"CSV聚合汇总 | {CV_PROTOCOL}")
        print("=" * 80)
        print(f"{'Run ID':>22} {'数据集':>15} {'模式':>10} {'Profile':>34} {'Folds':>7} {'SRCC':>14} {'PLCC':>14} {'KRCC':>14}")
        for result in results:
            profile = (
                f"{result.get('quality_preset', 'none')}/"
                f"{result.get('label_norm', 'none')}/"
                f"{result.get('rank_loss_type', 'hard')}/"
                f"balanced={result.get('use_balanced_sampling', False)}"
            )
            print(
                f"{str(result['run_id'])[-22:]:>22} {result['dataset']:>15} {result['ablation_mode']:>10} "
                f"{profile[-34:]:>34} "
                f"{result['num_folds']:>2}/{result['expected_folds']:<4} "
                f"{result['srcc_mean']:.4f}±{result['srcc_std']:.4f} "
                f"{result['plcc_mean']:.4f}±{result['plcc_std']:.4f} "
                f"{result['krcc_mean']:.4f}±{result['krcc_std']:.4f}"
            )

    return results


def analyze_cross_dataset_csv_dataframe(df, dataset=None, run_id=None):
    for required in ["source_dataset", "target_dataset", "srcc", "plcc", "krcc", "run_id"]:
        if required not in df.columns:
            if required == "run_id":
                df[required] = ""
            else:
                raise ValueError(f"跨库CSV缺少必要列: {required}")

    if dataset:
        df = df[(df["source_dataset"] == dataset) | (df["target_dataset"] == dataset)]
    if run_id:
        df = df[df["run_id"] == run_id]

    if df.empty:
        print("未找到匹配的跨库结果")
        return []

    for optional, default in [
        ("quality_preset", "none"),
        ("ablation_mode", "full"),
        ("epoch", ""),
        ("train_SRCC", 0.0),
        ("generalization_gap_srcc", 0.0),
        ("cv_protocol", "cross-dataset-full-test-folds"),
    ]:
        if optional not in df.columns:
            df[optional] = default
    df = df.rename(columns={"train_SRCC": "train_srcc"})

    results = []
    group_cols = ["run_id", "source_dataset", "target_dataset"]
    for (group_run_id, source, target), group in df.groupby(group_cols, dropna=False):
        group = group.sort_values("epoch").tail(1)
        row = group.iloc[0]
        result = {
            "run_id": group_run_id,
            "source_dataset": source,
            "target_dataset": target,
            "dataset_pair": f"{source}->{target}",
            "srcc": _safe_float(row.get("srcc")),
            "plcc": _safe_float(row.get("plcc")),
            "krcc": _safe_float(row.get("krcc")),
            "epoch": row.get("epoch", ""),
            "quality_preset": row.get("quality_preset", "none"),
            "ablation_mode": row.get("ablation_mode", "full"),
            "train_srcc": _safe_float(row.get("train_srcc", 0.0)),
            "generalization_gap_srcc": _safe_float(row.get("generalization_gap_srcc", 0.0)),
            "cv_protocol": row.get("cv_protocol", "cross-dataset-full-test-folds"),
        }
        results.append(result)

    print("\n" + "=" * 100)
    print("跨库实验CSV汇总 | cross-dataset-full-test-folds")
    print("=" * 100)
    print(f"{'Run ID':>22} {'Source':>15} {'Target':>15} {'Epoch':>6} {'SRCC':>8} {'PLCC':>8} {'KRCC':>8} {'TrainS':>8} {'Gap':>8} {'Preset':>12}")
    print("-" * 100)
    for result in results:
        print(
            f"{str(result['run_id'])[-22:]:>22} {result['source_dataset']:>15} {result['target_dataset']:>15} "
            f"{str(result['epoch']):>6} {result['srcc']:>8.4f} {result['plcc']:>8.4f} "
            f"{result['krcc']:>8.4f} {result['train_srcc']:>8.4f} "
            f"{result['generalization_gap_srcc']:>8.4f} {str(result['quality_preset']):>12}"
        )

    return results


def save_results_to_file(results, filename):
    if not isinstance(results, list):
        results = [results]

    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write(f"交叉验证结果分析报告 | {CV_PROTOCOL}\n")
        handle.write("=" * 80 + "\n\n")
        for result in results:
            if result is None:
                continue
            if "source_dataset" in result and "target_dataset" in result:
                handle.write(f"跨库: {result['source_dataset']} -> {result['target_dataset']}\n")
                handle.write(f"Run ID: {result.get('run_id', '')}\n")
                handle.write(f"Epoch: {result.get('epoch', '')}\n")
                handle.write(f"SRCC: {result['srcc']:.4f}\n")
                handle.write(f"PLCC: {result['plcc']:.4f}\n")
                handle.write(f"KRCC: {result['krcc']:.4f}\n\n")
            else:
                handle.write(f"数据集: {result['dataset']}\n")
                handle.write(f"Run ID: {result.get('run_id', '')}\n")
                handle.write(f"Fold状态: {result['num_folds']}/{result['expected_folds']} ({result['status']})\n")
                handle.write(f"SRCC: {result['srcc_mean']:.4f} ± {result['srcc_std']:.4f}\n")
                handle.write(f"PLCC: {result['plcc_mean']:.4f} ± {result['plcc_std']:.4f}\n")
                handle.write(f"KRCC: {result['krcc_mean']:.4f} ± {result['krcc_std']:.4f}\n\n")
        handle.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"结果已保存到: {filename}")


def main():
    parser = argparse.ArgumentParser(description="当前5折交叉验证结果分析器")
    parser.add_argument("--dataset", type=str, help="要分析的数据集名称")
    parser.add_argument("--ckpts_dir", type=str, default="ckpts", help="模型文件目录")
    parser.add_argument("--results_csv", type=str, default=None, help="从结果CSV按run_id聚合")
    parser.add_argument("--run_id", type=str, default=None, help="筛选指定run_id")
    parser.add_argument("--ablation_mode", type=str, default=None, help="筛选指定消融模式")
    parser.add_argument("--expected_folds", type=int, default=EXPECTED_FOLDS, help="当前协议期望fold数")
    parser.add_argument("--all", action="store_true", help="分析所有checkpoint数据集")
    parser.add_argument("--save", action="store_true", help="保存分析报告")
    args = parser.parse_args()

    if args.results_csv:
        results = analyze_results_csv(
            args.results_csv,
            dataset=args.dataset,
            run_id=args.run_id,
            ablation_mode=args.ablation_mode,
            expected_folds=args.expected_folds,
        )
    elif args.all:
        results = analyze_all_datasets(args.ckpts_dir, args.expected_folds)
    elif args.dataset:
        results = quick_analyze_with_csv_fallback(args.dataset, args.ckpts_dir, args.expected_folds)
    else:
        datasets = detect_available_datasets(args.ckpts_dir)
        if not datasets:
            print(f"未在 {args.ckpts_dir} 中检测到checkpoint")
            return
        print("检测到数据集:")
        for dataset in datasets:
            print(f"  - {dataset}")
        print("请使用 --dataset DATASET 或 --all 指定分析范围。")
        return

    if args.save and results:
        if isinstance(results, list):
            filename = "all_cv_results.txt"
        else:
            filename = f"{args.dataset}_cv_results.txt"
        save_results_to_file(results, filename)


if __name__ == "__main__":
    main()
