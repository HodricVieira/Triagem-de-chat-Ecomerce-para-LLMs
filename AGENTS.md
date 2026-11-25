# Regras e Contexto do Projeto — Triagem de Chat para Pequenos Negócios

## Objetivo e Escopo Atual
- O projeto investiga classificação de intenção em mensagens informais de atendimento em PT-BR (veja `Projeto.md` e `Trabalhos_Relacionados.md`).
- **Foco atual**: comparar o classificador **BERTimbau fine-tuned** (`modelo_intencao/`) contra **Gemini Flash-Lite (Zero-shot e Few-shot)** na tarefa de **classificação de intenção**, incluindo medição de **latência** e **projeção de custos em escala**.
- **Extração de entidades (NER)**: intencionalmente adiada para uma etapa futura; não implementar NER a menos que seja explicitamente solicitado.
- Ao iniciar um novo chat neste repositório, leia `CONTEXTO_PROJETO.md` para ver o status exato e os próximos passos.

## Arquitetura do Repositório e Fluxo Colab + GitHub + Drive
- **Pasta local**: `D:\rodri\Programacoes\Triagem-de-chat-Ecomerce-para-LLMs`
- **Repositório GitHub**: `https://github.com/HodricVieira/Triagem-de-chat-Ecomerce-para-LLMs`
- **Modelos fora do Git**: as pastas `modelo_intencao/`, `modelo_sentimento/` e `modelo_ner/` estão no `.gitignore` por serem pesadas demais para o GitHub.
- **Execução no Google Colab** (`Comparacao_BERTimbau_vs_LLM.ipynb`):
  - O código é sincronizado via GitHub (`git clone` / `git pull` em `/content/Triagem-de-chat-Ecomerce-para-LLMs`).
  - Os pesos salvos do BERTimbau de intenção são carregados do Google Drive (`MyDrive/Modelos_Triagem`).
  - Os caches incrementais (`*.jsonl`), gráficos e tabelas de saída são persistidos em `MyDrive/Modelos_Triagem/results` para sobreviver a reinícios da VM do Colab.

## Dataset e Classes Oficiais
- Dataset Hugging Face: `RichardSakaguchiMS/brazilian-customer-service-conversations` (`train`: 755, `validation`: 94, `test`: 95).
- Entrada de intenção: primeira mensagem da conversa (`ex["messages"][0]["content"]`); rótulo: `ex["metadata"]["intent"]`.
- **9 classes canônicas** (ordem alfabética do `LabelEncoder`):
  `cancelamento`, `compra`, `duvida_produto`, `duvida_servico`, `elogio`, `outros`, `reclamacao`, `saudacao`, `suporte_tecnico`.
- O split `test` (95 amostras) deve permanecer isolado para avaliação; exemplos *few-shot* devem sair exclusivamente do split `train`.

## Convenções de Ambiente e Segurança
- No Windows local, `python.exe` aponta para o stub da Microsoft Store e falha; execute scripts e testes via `uv run` (ex.: `uv run --offline --with pytest --with pydantic --with scikit-learn --with pandas --with numpy pytest -v tests/test_pipeline.py`).
- Nunca fixe caminhos absolutos do Windows (`C:\Users\...`) dentro de células destinadas ao Google Colab.
- Nunca versione `.env` ou chaves de API (`GEMINI_API_KEY`). Localmente use `.env` (baseado em `.env.example`); no Colab use **Colab Secrets** (`google.colab.userdata`) ou `.env` dentro de `MyDrive/Modelos_Triagem`.
- Mantenha `GEMINI_MODEL` configurável (família Flash-Lite); a Seção 2 de `Comparacao_BERTimbau_vs_LLM.ipynb` lista os IDs disponíveis na conta via `client.models.list()`.
