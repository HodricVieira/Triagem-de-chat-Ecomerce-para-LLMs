"""Avaliação via Gemini API (Zero-shot e Few-shot) com saída estruturada e cache JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from src.data_loader import INTENT_DESCRIPTIONS, INTENT_LABELS

IntentLiteral = Literal[
    "cancelamento",
    "compra",
    "duvida_produto",
    "duvida_servico",
    "elogio",
    "outros",
    "reclamacao",
    "saudacao",
    "suporte_tecnico",
]

OUTPUT_COLUMNS: list[str] = [
    "idx",
    "text",
    "pred_label",
    "latency_ms",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "status",
    "error",
]


class IntentPrediction(BaseModel):
    """Schema estruturado garantindo que o LLM retorne uma das 9 intenções válidas."""

    intent: IntentLiteral = Field(
        description="Rótulo exato da intenção principal da mensagem do cliente."
    )


def build_system_instruction() -> str:
    """Constrói a instrução de sistema em PT-BR descrevendo as 9 classes de intenção."""
    classes_block = "\n".join(
        f"- {label}: {INTENT_DESCRIPTIONS[label]}" for label in INTENT_LABELS
    )
    return (
        "Você é um sistema de triagem automática de mensagens informais de atendimento "
        "ao cliente em português brasileiro (gírias, abreviações, erros de digitação e "
        "ausência de pontuação).\n"
        "Sua tarefa é classificar a mensagem do cliente em EXATAMENTE UMA das "
        "intenções abaixo:\n"
        f"{classes_block}\n\n"
        "Responda estritamente no formato JSON solicitado, escolhendo apenas um dos "
        "rótulos exatos listados acima."
    )


def build_user_prompt(
    text: str,
    few_shot_examples: Sequence[dict[str, str]] | None = None,
) -> str:
    """Formata o prompt do usuário para inferência zero-shot ou few-shot."""
    if few_shot_examples:
        formatted_examples = "\n".join(
            f'{idx}. Mensagem: "{ex["text"]}" -> Intenção: {ex["intent"]}'
            for idx, ex in enumerate(few_shot_examples, start=1)
        )
        return (
            "Considere os exemplos rotulados abaixo como referência:\n"
            f"{formatted_examples}\n\n"
            "Agora classifique a nova mensagem abaixo:\n"
            f'Mensagem do cliente: "{text}"'
        )

    return (
        "Classifique a intenção da mensagem abaixo:\n"
        f'Mensagem do cliente: "{text}"'
    )


def _load_cache(cache_path: Path) -> dict[int, dict[str, Any]]:
    """Carrega registros concluídos com sucesso de um arquivo JSONL."""
    cached: dict[int, dict[str, Any]] = {}
    if not cache_path.exists():
        return cached

    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok" and "idx" in record:
                cached[int(record["idx"])] = record
    return cached


def _append_cache(cache_path: Path, record: dict[str, Any]) -> None:
    """Acrescenta uma predição concluída ao arquivo de cache JSONL."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_transient_error(exc: Exception) -> bool:
    """Identifica erros transitórios de rede ou limite de taxa elegíveis para retry."""
    msg = str(exc).lower()
    transient_markers = (
        "429",
        "503",
        "500",
        "resource_exhausted",
        "resourceexhausted",
        "rate",
        "quota",
        "unavailable",
        "timeout",
        "timed out",
        "connection",
    )
    return any(marker in msg for marker in transient_markers)


def _mock_predict(
    idx: int,
    text: str,
    system_instruction: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Gera predição determinística offline para testes sem chamadas de rede."""
    digest = hashlib.md5(text.encode("utf-8")).digest()
    pred_label = INTENT_LABELS[digest[0] % len(INTENT_LABELS)]
    prompt_words = len(system_instruction.split()) + len(user_prompt.split())
    prompt_tokens = max(20, int(prompt_words * 1.4))
    output_tokens = 8
    return {
        "idx": idx,
        "text": text,
        "pred_label": pred_label,
        "latency_ms": 110.0 + float(digest[1] % 35),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "status": "ok",
        "error": None,
    }


def _call_gemini_with_retry(
    client: Any,
    types_module: Any,
    model_name: str,
    system_instruction: str,
    user_prompt: str,
    max_retries: int,
) -> tuple[str, float, int, int, int]:
    """Executa uma chamada ao Gemini com backoff exponencial para falhas transitórias."""
    config = types_module.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=IntentPrediction,
    )

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            start = time.perf_counter()
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0

            parsed = IntentPrediction.model_validate_json(response.text)
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            total_tokens = int(
                getattr(usage, "total_token_count", 0) or (prompt_tokens + output_tokens)
            )
            return parsed.intent, latency_ms, prompt_tokens, output_tokens, total_tokens
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_retries or not _is_transient_error(exc):
                raise
            sleep_s = (2**attempt) + random.uniform(0.1, 0.6)
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


def evaluate_gemini(
    texts: Sequence[str],
    mode: Literal["zero_shot", "few_shot"] = "zero_shot",
    model_name: str = "gemini-3.1-flash-lite-preview",
    few_shot_examples: Sequence[dict[str, str]] | None = None,
    api_key: str | None = None,
    cache_path: str | Path | None = None,
    max_retries: int = 4,
    request_delay_s: float = 0.0,
    mock: bool = False,
) -> pd.DataFrame:
    """Avalia uma lista de mensagens com Gemini Flash-Lite em modo zero_shot ou few_shot."""
    if mode not in ("zero_shot", "few_shot"):
        raise ValueError(f"Modo inválido: '{mode}'. Use 'zero_shot' ou 'few_shot'.")
    if mode == "few_shot" and not few_shot_examples:
        raise ValueError("few_shot_examples deve ser fornecido quando mode='few_shot'.")

    if not texts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    active_examples = few_shot_examples if mode == "few_shot" else None
    system_instruction = build_system_instruction()

    resolved_cache_path = Path(cache_path) if cache_path is not None else None
    cached_records = _load_cache(resolved_cache_path) if resolved_cache_path else {}

    client: Any = None
    types_module: Any = None
    if not mock:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY não configurada. Defina a variável de ambiente ou passe api_key."
            )
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=resolved_key)
        types_module = genai_types

    records: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        cached = cached_records.get(idx)
        if cached is not None and cached.get("text") == text:
            records.append(cached)
            continue

        user_prompt = build_user_prompt(text, few_shot_examples=active_examples)

        if mock:
            record = _mock_predict(idx, text, system_instruction, user_prompt)
        else:
            try:
                (
                    pred_label,
                    latency_ms,
                    prompt_tokens,
                    output_tokens,
                    total_tokens,
                ) = _call_gemini_with_retry(
                    client=client,
                    types_module=types_module,
                    model_name=model_name,
                    system_instruction=system_instruction,
                    user_prompt=user_prompt,
                    max_retries=max_retries,
                )
                record = {
                    "idx": idx,
                    "text": text,
                    "pred_label": pred_label,
                    "latency_ms": latency_ms,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "status": "ok",
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                record = {
                    "idx": idx,
                    "text": text,
                    "pred_label": None,
                    "latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "status": "error",
                    "error": str(exc),
                }

            if request_delay_s > 0:
                time.sleep(request_delay_s)

        if resolved_cache_path is not None and record["status"] == "ok":
            _append_cache(resolved_cache_path, record)

        records.append(record)

    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    return df.sort_values("idx").reset_index(drop=True)
