"""Cálculo de métricas de classificação, latência, consumo de tokens e projeção de custos."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import INTENT_LABELS

DEFAULT_HOSTING_SCENARIOS: dict[str, float] = {
    "Local (custo marginal)": 0.0,
    "VPS CPU básica (US$ 10/mês)": 10.0,
    "Instância dedicada (US$ 40/mês)": 40.0,
}


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    labels: Sequence[str] = INTENT_LABELS,
) -> dict[str, Any]:
    """Calcula métricas globais, relatório por classe e matriz de confusão."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo comprimento.")

    label_list = list(labels)
    y_true_clean = [str(item) for item in y_true]
    y_pred_clean = [str(item) if item is not None else "__erro__" for item in y_pred]

    accuracy = float(accuracy_score(y_true_clean, y_pred_clean))
    precision_macro = float(
        precision_score(
            y_true_clean, y_pred_clean, labels=label_list, average="macro", zero_division=0
        )
    )
    recall_macro = float(
        recall_score(
            y_true_clean, y_pred_clean, labels=label_list, average="macro", zero_division=0
        )
    )
    f1_macro = float(
        f1_score(
            y_true_clean, y_pred_clean, labels=label_list, average="macro", zero_division=0
        )
    )
    f1_weighted = float(
        f1_score(
            y_true_clean, y_pred_clean, labels=label_list, average="weighted", zero_division=0
        )
    )

    report_dict = classification_report(
        y_true_clean,
        y_pred_clean,
        labels=label_list,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true_clean,
        y_pred_clean,
        labels=label_list,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=label_list).tolist()

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
    }


def summarize_latency_and_tokens(df: pd.DataFrame) -> dict[str, float]:
    """Resume latência (média, p50, p95) e consumo médio de tokens de um DataFrame de avaliação."""
    if df.empty:
        return {
            "latency_mean_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_total_tokens": 0.0,
        }

    valid_df = df[df["status"] == "ok"] if "status" in df.columns else df
    if valid_df.empty:
        valid_df = df

    latencies = valid_df["latency_ms"].astype(float).to_numpy()
    avg_prompt = (
        float(valid_df["prompt_tokens"].astype(float).mean())
        if "prompt_tokens" in valid_df.columns
        else 0.0
    )
    avg_output = (
        float(valid_df["output_tokens"].astype(float).mean())
        if "output_tokens" in valid_df.columns
        else 0.0
    )
    avg_total = (
        float(valid_df["total_tokens"].astype(float).mean())
        if "total_tokens" in valid_df.columns
        else 0.0
    )

    return {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "avg_prompt_tokens": avg_prompt,
        "avg_output_tokens": avg_output,
        "avg_total_tokens": avg_total,
    }


def build_comparison_table(model_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Consolida métricas de qualidade, latência e tokens de múltiplos modelos em uma tabela."""
    columns = [
        "Model",
        "Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1 (Macro)",
        "F1 (Weighted)",
        "Latency Mean (ms)",
        "Latency P95 (ms)",
        "Avg Prompt Tokens",
        "Avg Output Tokens",
    ]

    rows: list[dict[str, Any]] = []
    for model_name, payload in model_results.items():
        metrics = payload.get("metrics", payload)
        stats = payload.get("stats", payload)
        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(float(metrics["accuracy"]), 4),
                "Precision (Macro)": round(float(metrics["precision_macro"]), 4),
                "Recall (Macro)": round(float(metrics["recall_macro"]), 4),
                "F1 (Macro)": round(float(metrics["f1_macro"]), 4),
                "F1 (Weighted)": round(float(metrics["f1_weighted"]), 4),
                "Latency Mean (ms)": round(float(stats.get("latency_mean_ms", 0.0)), 2),
                "Latency P95 (ms)": round(float(stats.get("latency_p95_ms", 0.0)), 2),
                "Avg Prompt Tokens": round(float(stats.get("avg_prompt_tokens", 0.0)), 1),
                "Avg Output Tokens": round(float(stats.get("avg_output_tokens", 0.0)), 1),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def project_monthly_costs(
    avg_prompt_tokens: float,
    avg_output_tokens: float,
    input_price_per_1m: float = 0.10,
    output_price_per_1m: float = 0.40,
    volumes: Sequence[int] = (1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000),
    local_hosting_scenarios: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Projeta custos mensais da API vs. hospedagem do BERTimbau e calcula ponto de break-even."""
    scenarios = (
        dict(local_hosting_scenarios)
        if local_hosting_scenarios is not None
        else dict(DEFAULT_HOSTING_SCENARIOS)
    )

    cost_per_message_usd = (
        avg_prompt_tokens * input_price_per_1m + avg_output_tokens * output_price_per_1m
    ) / 1_000_000.0

    rows: list[dict[str, Any]] = []
    for volume in volumes:
        api_monthly_cost = float(volume) * cost_per_message_usd
        row: dict[str, Any] = {
            "volume_mensal": int(volume),
            "custo_por_msg_usd": cost_per_message_usd,
            "api_custo_mensal_usd": round(api_monthly_cost, 4),
        }
        for scenario_name, fixed_cost in scenarios.items():
            row[f"{scenario_name} (USD)"] = round(float(fixed_cost), 2)
            if api_monthly_cost > 0:
                savings_pct = ((api_monthly_cost - fixed_cost) / api_monthly_cost) * 100.0
            else:
                savings_pct = 0.0
            row[f"Economia vs {scenario_name} (%)"] = round(savings_pct, 2)
        rows.append(row)

    breakeven_messages: dict[str, float | None] = {}
    for scenario_name, fixed_cost in scenarios.items():
        if cost_per_message_usd <= 0:
            breakeven_messages[scenario_name] = None
        elif fixed_cost <= 0:
            breakeven_messages[scenario_name] = 0.0
        else:
            breakeven_messages[scenario_name] = float(fixed_cost / cost_per_message_usd)

    return pd.DataFrame(rows), breakeven_messages
