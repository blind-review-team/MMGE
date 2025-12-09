import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from datetime import datetime


def load_experiment_result(file_path):
    """加载实验结果文件

    Args:
        file_path (str): 实验结果JSON文件路径

    Returns:
        dict: 加载的实验结果
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"实验结果文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    return result


def extract_best_result(result):
    """从实验结果中提取最佳结果

    Args:
        result (dict): 实验结果字典

    Returns:
        dict: 最佳结果
    """
    if not result["experiments"]:
        return None

    # 默认按照Recall@20寻找最佳结果
    best_exp = None
    best_value = -1

    for exp in result["experiments"]:
        if "test_result" in exp and "recall@20" in exp["test_result"]:
            if exp["test_result"]["recall@20"] > best_value:
                best_value = exp["test_result"]["recall@20"]
                best_exp = exp

    return best_exp


def format_metric_name(metric_name):
    """格式化指标名称，使其更易读

    Args:
        metric_name (str): 原始指标名称

    Returns:
        str: 格式化后的指标名称
    """
    if '@' in metric_name:
        base, k = metric_name.split('@')
        return f"{base.upper()}@{k}"
    return metric_name.upper()


def compare_experiment_results(result_files, metrics=None, output_dir=None):
    """比较多个实验结果

    Args:
        result_files (list): 实验结果文件路径列表
        metrics (list, optional): 要比较的指标列表，默认为None（比较所有指标）
        output_dir (str, optional): 输出结果的目录，默认为None（使用当前目录）

    Returns:
        pd.DataFrame: 比较结果
    """
    if not result_files:
        print("未提供实验结果文件")
        return None

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "result_comparisons")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载所有实验结果
    results = []
    for file_path in result_files:
        try:
            result = load_experiment_result(file_path)
            results.append(result)
            print(f"已加载实验结果: {file_path}")
        except Exception as e:
            print(f"加载实验结果失败 {file_path}: {e}")

    if not results:
        print("未能加载任何实验结果")
        return None

    # 准备比较数据
    comparison_data = []

    for result in results:
        best_exp = extract_best_result(result)
        if not best_exp:
            continue

        row = {
            "model": result.get("model", "未知"),
            "dataset": result.get("dataset", "未知"),
            "timestamp": result.get("timestamp", "未知"),
            "description": result.get("description", ""),
        }

        # 添加参数
        for param_name, param_value in best_exp.get("parameters", {}).items():
            row[f"param_{param_name}"] = param_value

        # 添加测试结果
        for metric_name, metric_value in best_exp.get("test_result", {}).items():
            row[f"test_{metric_name}"] = metric_value

        comparison_data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(comparison_data)

    # 过滤指标（如果指定）
    if metrics:
        filtered_columns = [col for col in df.columns if any(col.endswith(m) for m in metrics)]
        filtered_columns = ["model", "dataset", "description"] + filtered_columns
        df = df[filtered_columns]

    # 保存比较结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"comparison_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    # 打印比较表格
    print("\n实验结果比较:")
    display_columns = [
        "model", "dataset", "description",
        *[col for col in df.columns if col.startswith("test_recall@")],
        *[col for col in df.columns if col.startswith("test_ndcg@")]
    ]
    display_df = df[display_columns].copy()

    # 格式化列名
    display_df.columns = [col.replace("test_", "") if col.startswith("test_") else col for col in display_df.columns]

    # 找出每列的最大值并高亮（仅对指标列）
    for col in display_df.columns:
        if col not in ["model", "dataset", "description"]:
            display_df[col] = display_df[col].round(4)

    print(tabulate(display_df, headers="keys", tablefmt="grid"))
    print(f"\n比较结果已保存到: {csv_path}")

    # 绘制对比图
    plot_metric_comparison(df, output_dir, timestamp)

    return df


def plot_metric_comparison(df, output_dir, timestamp):
    """绘制指标对比图

    Args:
        df (pd.DataFrame): 比较结果DataFrame
        output_dir (str): 输出目录
        timestamp (str): 时间戳
    """
    # 提取Recall和NDCG指标
    recall_cols = [col for col in df.columns if col.startswith("test_recall@")]
    ndcg_cols = [col for col in df.columns if col.startswith("test_ndcg@")]

    if not recall_cols and not ndcg_cols:
        return

    # 为每个实验创建标签
    labels = []
    for i, row in df.iterrows():
        if row.get("description"):
            label = f"{row['model']} - {row['description']}"
        else:
            label = f"{row['model']} ({i + 1})"
        labels.append(label)

    # 绘制Recall对比图
    if recall_cols:
        plt.figure(figsize=(10, 6))
        x_labels = [col.replace("test_recall@", "") for col in recall_cols]

        for i, row in df.iterrows():
            values = [row[col] for col in recall_cols]
            plt.plot(x_labels, values, marker='o', label=labels[i])

        plt.title("Recall@K对比")
        plt.xlabel("K值")
        plt.ylabel("Recall")
        plt.legend()
        plt.grid(True)
        recall_plot_path = os.path.join(output_dir, f"recall_comparison_{timestamp}.png")
        plt.savefig(recall_plot_path)
        plt.close()
        print(f"Recall对比图已保存到: {recall_plot_path}")

    # 绘制NDCG对比图
    if ndcg_cols:
        plt.figure(figsize=(10, 6))
        x_labels = [col.replace("test_ndcg@", "") for col in ndcg_cols]

        for i, row in df.iterrows():
            values = [row[col] for col in ndcg_cols]
            plt.plot(x_labels, values, marker='o', label=labels[i])

        plt.title("NDCG@K对比")
        plt.xlabel("K值")
        plt.ylabel("NDCG")
        plt.legend()
        plt.grid(True)
        ndcg_plot_path = os.path.join(output_dir, f"ndcg_comparison_{timestamp}.png")
        plt.savefig(ndcg_plot_path)
        plt.close()
        print(f"NDCG对比图已保存到: {ndcg_plot_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分析和比较实验结果")
    parser.add_argument("result_files", nargs="+", help="实验结果文件路径列表")
    parser.add_argument("--metrics", nargs="+", help="要比较的指标列表")
    parser.add_argument("--output_dir", help="输出结果的目录")

    args = parser.parse_args()

    compare_experiment_results(args.result_files, args.metrics, args.output_dir) 