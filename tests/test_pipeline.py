"""Testes unitários offline para o pipeline de comparação BERTimbau vs. Gemini."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import INTENT_LABELS, select_few_shot_examples
from src.llm_eval import OUTPUT_COLUMNS, evaluate_gemini
from src.metrics_and_cost import (
    build_comparison_table,
    compute_classification_metrics,
    project_monthly_costs,
    summarize_latency_and_tokens,
)


def _make_synthetic_train(n_per_class: int = 3) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for label in INTENT_LABELS:
        for i in range(n_per_class):
            texts.append(f"mensagem de treino {label} exemplo {i}")
            labels.append(label)
    return texts, labels


def test_select_few_shot_examples_stratified_and_deterministic() -> None:
    train_texts, train_labels = _make_synthetic_train(n_per_class=4)

    first = select_few_shot_examples(train_texts, train_labels, k_per_class=2, seed=42)
    second = select_few_shot_examples(train_texts, train_labels, k_per_class=2, seed=42)

    assert len(first) == len(INTENT_LABELS) * 2
    assert first == second
    assert {item["intent"] for item in first} == set(INTENT_LABELS)
    assert all(item["text"] in train_texts for item in first)


def test_select_few_shot_examples_validation_errors() -> None:
    train_texts, train_labels = _make_synthetic_train(n_per_class=1)

    with pytest.raises(ValueError, match="pelo menos 2"):
        select_few_shot_examples(train_texts, train_labels, k_per_class=2)

    with pytest.raises(ValueError, match="mesmo comprimento"):
        select_few_shot_examples(train_texts[:-1], train_labels, k_per_class=1)

    with pytest.raises(ValueError, match="maior ou igual a 1"):
        select_few_shot_examples(train_texts, train_labels, k_per_class=0)


def test_evaluate_gemini_mock_tokens_and_cache_reuse(tmp_path: Path) -> None:
    train_texts, train_labels = _make_synthetic_train(n_per_class=2)
    few_shot_examples = select_few_shot_examples(train_texts, train_labels, k_per_class=1, seed=7)
    test_texts = [
        "bom dia gostaria de saber o preco",
        "quero cancelar meu plano agora",
        "obrigado pelo otimo atendimento",
    ]

    zs_cache = tmp_path / "zs_cache.jsonl"
    df_zs = evaluate_gemini(
        test_texts,
        mode="zero_shot",
        cache_path=zs_cache,
        mock=True,
    )
    df_fs = evaluate_gemini(
        test_texts,
        mode="few_shot",
        few_shot_examples=few_shot_examples,
        cache_path=tmp_path / "fs_cache.jsonl",
        mock=True,
    )

    assert list(df_zs.columns) == OUTPUT_COLUMNS
    assert len(df_zs) == len(test_texts)
    assert set(df_zs["pred_label"]).issubset(set(INTENT_LABELS))
    assert (df_zs["status"] == "ok").all()
    assert df_fs["prompt_tokens"].mean() > df_zs["prompt_tokens"].mean()

    # Garante que uma segunda execução reutiliza o cache sem duplicar linhas no JSONL
    lines_before = zs_cache.read_text(encoding="utf-8").splitlines()
    assert len(lines_before) == len(test_texts)

    # Marca um registro no cache com rótulo específico para comprovar reuso real
    modified_first = json.loads(lines_before[0])
    modified_first["pred_label"] = "elogio"
    zs_cache.write_text(
        "\n".join([json.dumps(modified_first, ensure_ascii=False), *lines_before[1:]]) + "\n",
        encoding="utf-8",
    )

    df_zs_cached = evaluate_gemini(
        test_texts,
        mode="zero_shot",
        cache_path=zs_cache,
        mock=True,
    )
    lines_after = zs_cache.read_text(encoding="utf-8").splitlines()
    assert len(lines_after) == len(test_texts)
    assert df_zs_cached.loc[0, "pred_label"] == "elogio"


def test_evaluate_gemini_requires_examples_for_few_shot() -> None:
    with pytest.raises(ValueError, match="few_shot_examples"):
        evaluate_gemini(["oi"], mode="few_shot", mock=True)


def test_metrics_and_comparison_table() -> None:
    y_true = list(INTENT_LABELS)
    y_pred = list(INTENT_LABELS)
    y_pred[0] = "reclamacao"  # 8/9 corretas

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(8 / 9)
    assert len(metrics["confusion_matrix"]) == len(INTENT_LABELS)
    assert metrics["confusion_matrix"][0][INTENT_LABELS.index("reclamacao")] == 1

    import pandas as pd

    df_bert = pd.DataFrame(
        {
            "idx": range(len(y_true)),
            "text": ["msg"] * len(y_true),
            "pred_label": y_pred,
            "confidence": [0.95] * len(y_true),
            "latency_ms": [12.0] * len(y_true),
        }
    )
    bert_stats = summarize_latency_and_tokens(df_bert)
    assert bert_stats["latency_mean_ms"] == pytest.approx(12.0)
    assert bert_stats["avg_prompt_tokens"] == 0.0

    table = build_comparison_table(
        {"BERTimbau (Fine-tuned)": {"metrics": metrics, "stats": bert_stats}}
    )
    expected_cols = [
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
    assert list(table.columns) == expected_cols
    assert table.loc[0, "Accuracy"] == pytest.approx(round(8 / 9, 4))


def test_project_monthly_costs_and_breakeven() -> None:
    # 500 input tokens * $0.10/1M + 10 output tokens * $0.40/1M = $0.000054/msg
    df_cost, breakeven = project_monthly_costs(
        avg_prompt_tokens=500.0,
        avg_output_tokens=10.0,
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
        volumes=(100_000, 1_000_000),
        local_hosting_scenarios={"Local": 0.0, "VPS": 10.0},
    )

    assert df_cost.loc[0, "custo_por_msg_usd"] == pytest.approx(0.000054)
    assert df_cost.loc[0, "api_custo_mensal_usd"] == pytest.approx(5.4)
    assert df_cost.loc[1, "api_custo_mensal_usd"] == pytest.approx(54.0)
    # Em 1M msgs: API=$54 vs VPS=$10 -> economia = (54-10)/54 * 100 = 81.48%
    assert df_cost.loc[1, "Economia vs VPS (%)"] == pytest.approx(81.48, abs=0.01)
    assert breakeven["Local"] == 0.0
    assert breakeven["VPS"] == pytest.approx(10.0 / 0.000054)

    _, breakeven_zero = project_monthly_costs(
        avg_prompt_tokens=0.0,
        avg_output_tokens=0.0,
        volumes=(1_000,),
        local_hosting_scenarios={"VPS": 10.0},
    )
    assert breakeven_zero["VPS"] is None


def test_notebook_json_and_python_syntax_valid() -> None:
    nb_path = REPO_ROOT / "Comparacao_BERTimbau_vs_LLM.ipynb"
    assert nb_path.exists()

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            compile(code, f"Comparacao_BERTimbau_vs_LLM.ipynb:cell_{idx}", "exec")
