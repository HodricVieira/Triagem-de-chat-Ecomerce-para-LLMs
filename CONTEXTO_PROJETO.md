# Contexto de Continuidade do Projeto (`CONTEXTO_PROJETO.md`)

> Referência da conversa anterior: [Conversa anterior](conversation://5b82e1c2-d975-4b03-b434-a8202cbe4008)

## 1. O que já foi implementado

### Modelos já treinados localmente (`Bertimbau_costumer_service.ipynb`)
- **`modelo_intencao/`** (`neuralmind/bert-base-portuguese-cased` fine-tuned em 3 épocas):
  - 9 classes de intenção (`intent_encoder.joblib`).
  - Desempenho observado no split `validation` (94 amostras): **87,23% de acurácia**, **0,86 F1-macro**, **0,87 F1-weighted**.
- **`modelo_sentimento/`** (3 classes: `negative`, `neutral`, `positive`):
  - Desempenho observado no split `validation` (94 amostras): **90,43% de acurácia**, **0,90 F1-macro**.

### Pipeline comparativo BERTimbau vs. Gemini Flash-Lite
- `src/__init__.py`: inicialização do pacote.
- `src/data_loader.py`: define `INTENT_LABELS`, `INTENT_DESCRIPTIONS`, `load_intent_splits()` e `select_few_shot_examples()` (amostragem estratificada determinística exclusivamente do `train`).
- `src/bertimbau_eval.py`: carrega `modelo_intencao/` em GPU ou CPU (`load_bertimbau_intent_model`) e avalia mensagem a mensagem com warm-up e medição de latência (`evaluate_bertimbau`).
- `src/llm_eval.py`: avaliador via SDK `google-genai` com saída JSON estruturada via Pydantic (`IntentPrediction`), modos `zero_shot` e `few_shot`, *retry* com backoff exponencial, cache incremental em JSONL (`_load_cache` / `_append_cache`) e modo offline `mock=True`.
- `src/metrics_and_cost.py`: métricas completas (`compute_classification_metrics`), resumo de latência/tokens (`summarize_latency_and_tokens`), tabela consolidada (`build_comparison_table`) e projeção matemática de custos mensais + *break-even* (`project_monthly_costs`).
- `Comparacao_BERTimbau_vs_LLM.ipynb`: notebook comparativo já adaptado para o fluxo híbrido **GitHub + Google Drive** no Colab:
  - Clona/atualiza `https://github.com/HodricVieira/Triagem-de-chat-Ecomerce-para-LLMs.git`.
  - Monta o Google Drive e localiza `intent_encoder.joblib` dentro de `MyDrive/Modelos_Triagem` (`MODEL_INTENT_DIR`).
  - Salva caches e artefatos finais (`*.png`, `*.csv`, `comparison_summary.json`) em `MyDrive/Modelos_Triagem/results`.
- `requirements.txt` e `.env.example`.
- `tests/test_pipeline.py`: 7 testes unitários offline (todos passaram antes da renomeação da pasta local; precisam ser reexecutados nesta pasta para confirmar o notebook atualizado).

---

## 2. Estado Atual e Próximos Passos Imediatos

1. **Publicar alterações pendentes no GitHub** (a partir de `D:\rodri\Programacoes\Triagem-de-chat-Ecomerce-para-LLMs`):
   - Verificar `git status` e fazer `git add`, `git commit` e `git push` para que o Colab puxe a versão mais recente de `Comparacao_BERTimbau_vs_LLM.ipynb`, `src/`, `AGENTS.md` e `CONTEXTO_PROJETO.md`.
2. **Validar testes unitários na pasta atual**:
   ```powershell
   uv run --offline --with pytest --with pydantic --with scikit-learn --with pandas --with numpy pytest -v tests/test_pipeline.py
   ```
   *(Se algum pacote não estiver em cache, rodar o mesmo comando sem `--offline`.)*
3. **Configurar a chave da API no Google Colab**:
   - Adicionar `GEMINI_API_KEY` nos **Secrets** do Colab (com acesso habilitado ao notebook) ou salvar um `.env` em `MyDrive/Modelos_Triagem/.env`.
4. **Executar o experimento real no Google Colab**:
   - Rodar a Seção 1 (`cell-bootstrap` + parâmetros) e conferir se `src/data_loader.py` e `MODEL_INTENT_DIR` foram encontrados.
   - Rodar a Seção 2 (`cell-model-check`) para confirmar o ID exato do modelo Flash-Lite habilitado na conta (ex.: `gemini-3.1-flash-lite-preview`, `gemini-2.5-flash-lite` ou `gemini-3.5-flash-lite`) e ajustar `GEMINI_MODEL` se necessário.
   - Executar as Seções 3 a 9 sobre as 95 amostras do split `test` e inspecionar os resultados salvos em `MyDrive/Modelos_Triagem/results/`.

---

## 3. Escopo Futuro (Adiado Propositalmente)
- Extração de entidades (NER) com saída JSON estruturada.
- Avaliação de robustez a ruído sintético e compilação/teste com 50–100 mensagens reais anonimizadas (`Projeto.md`).
- Baseline clássico (TF-IDF + Regressão Logística/SVM), se necessário para fechar a comparação do artigo/relatório.
