"""Carregamento do dataset de atendimento em PT-BR e seleção de exemplos few-shot."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

INTENT_LABELS: tuple[str, ...] = (
    "cancelamento",
    "compra",
    "duvida_produto",
    "duvida_servico",
    "elogio",
    "outros",
    "reclamacao",
    "saudacao",
    "suporte_tecnico",
)

INTENT_DESCRIPTIONS: dict[str, str] = {
    "cancelamento": "Cliente solicita cancelar um pedido, assinatura, contrato ou serviço.",
    "compra": "Cliente demonstra intenção explícita de comprar, contratar, fechar pedido ou pede link/forma de pagamento.",
    "duvida_produto": "Cliente pergunta sobre características, preço, estoque ou disponibilidade de um produto.",
    "duvida_servico": "Cliente pergunta sobre como funciona um serviço, prazos, planos, horários ou agendamento.",
    "elogio": "Cliente expressa satisfação, agradecimento ou elogio ao atendimento, produto ou serviço.",
    "outros": "Mensagens fora das demais categorias (assuntos gerais, confirmações simples ou contexto ambíguo).",
    "reclamacao": "Cliente relata insatisfação, atraso, cobrança indevida, defeito ou atendimento ruim sem pedir suporte passo a passo nem cancelamento.",
    "saudacao": "Mensagem inicial apenas de cumprimento ou abertura de conversa (ex.: 'oi', 'bom dia'), sem solicitação específica.",
    "suporte_tecnico": "Cliente relata falha técnica ou pede ajuda/instruções para configurar, acessar ou resolver erro operacional.",
}


def load_intent_splits(
    dataset_name: str = "RichardSakaguchiMS/brazilian-customer-service-conversations",
) -> dict[str, tuple[list[str], list[str]]]:
    """Carrega os splits train, validation e test extraindo a primeira mensagem e a intenção."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    splits: dict[str, tuple[list[str], list[str]]] = {}

    for split_name in ("train", "validation", "test"):
        split_data = dataset[split_name]
        texts = [ex["messages"][0]["content"] for ex in split_data]
        labels = [ex["metadata"]["intent"] for ex in split_data]
        splits[split_name] = (texts, labels)

    return splits


def select_few_shot_examples(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    k_per_class: int = 1,
    seed: int = 42,
    labels: Sequence[str] = INTENT_LABELS,
) -> list[dict[str, str]]:
    """Seleciona de forma determinística e estratificada k exemplos por classe apenas do split train."""
    if len(train_texts) != len(train_labels):
        raise ValueError("train_texts e train_labels devem ter o mesmo comprimento.")
    if k_per_class < 1:
        raise ValueError("k_per_class deve ser maior ou igual a 1.")

    grouped: dict[str, list[str]] = defaultdict(list)
    for text, label in zip(train_texts, train_labels):
        grouped[label].append(text)

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []

    for label in labels:
        candidates = grouped.get(label, [])
        if len(candidates) < k_per_class:
            raise ValueError(
                f"Classe '{label}' possui apenas {len(candidates)} exemplos no treino; "
                f"são necessários pelo menos {k_per_class}."
            )
        chosen_texts = rng.sample(candidates, k_per_class)
        for text in chosen_texts:
            selected.append({"text": text, "intent": label})

    return selected
