"""Carregamento e avaliação do modelo BERTimbau fine-tuned para classificação de intenção."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def load_bertimbau_intent_model(
    model_dir: str | Path = "modelo_intencao",
    device: str | None = None,
) -> tuple[Any, Any, Any, str]:
    """Carrega o modelo BERTimbau, tokenizador e LabelEncoder salvos em model_dir."""
    import joblib
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Diretório do modelo não encontrado: {model_path}")

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    encoder = joblib.load(model_path / "intent_encoder.joblib")

    model.to(resolved_device)
    model.eval()
    return model, tokenizer, encoder, resolved_device


def _infer_single(
    text: str,
    model: Any,
    tokenizer: Any,
    encoder: Any,
    resolved_device: str,
    max_length: int,
    torch_module: Any,
) -> tuple[str, float]:
    """Executa inferência em uma única mensagem e retorna (classe_predita, confiança)."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(resolved_device) for k, v in inputs.items()}

    with torch_module.no_grad():
        outputs = model(**inputs)
        probs = torch_module.softmax(outputs.logits, dim=-1)
        pred_id = int(torch_module.argmax(probs, dim=-1).item())
        confidence = float(probs[0, pred_id].item())

    pred_label = str(encoder.inverse_transform([pred_id])[0])
    return pred_label, confidence


def evaluate_bertimbau(
    texts: Sequence[str],
    model: Any,
    tokenizer: Any,
    encoder: Any,
    device: str | None = None,
    max_length: int = 128,
    warmup_runs: int = 3,
) -> pd.DataFrame:
    """Avalia mensagens individualmente medindo predição, confiança e latência (ms)."""
    import torch

    columns = ["idx", "text", "pred_label", "confidence", "latency_ms"]
    if not texts:
        return pd.DataFrame(columns=columns)

    resolved_device = device or str(getattr(model, "device", "cpu"))
    use_cuda_sync = resolved_device.startswith("cuda") and torch.cuda.is_available()

    for _ in range(max(0, warmup_runs)):
        _infer_single(texts[0], model, tokenizer, encoder, resolved_device, max_length, torch)

    rows: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        if use_cuda_sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred_label, confidence = _infer_single(
            text, model, tokenizer, encoder, resolved_device, max_length, torch
        )
        if use_cuda_sync:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        rows.append(
            {
                "idx": idx,
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "latency_ms": latency_ms,
            }
        )

    return pd.DataFrame(rows, columns=columns)
