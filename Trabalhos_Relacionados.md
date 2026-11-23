# **Trabalhos Relacionados**

## **Metodologia de Busca**

A pesquisa bibliográfica foi realizada utilizando os repositórios acadêmicos Google Scholar, arXiv e ACL Anthology. As queries de busca incluíram combinações de termos como: "intent classification informal text", "named entity recognition Portuguese social media", "BERTimbau fine-tuning", e "LLM vs Encoder models cost efficiency". Considerou-se o período de 2018 a 2026, foram inicialmente considerados 13 artigos com o objetivo de capturar o estado da arte em modelos baseados em Transformers, SLMs e LLMs. Foi selecionado 1 artigo sobre o estado da arte do BERTimbau, 2 artigos de comparação entre modelos BERT e LLMs, 1 artigo sobre BERT para E-commerce e 1 artigo sobre textos ruidosos em modelos de IA.

## **Trabalhos Relacionados**

Souza et al. (2020) introduziram o BERTimbau, demonstrando resultados no estado da arte para tarefas de Reconhecimento de Entidades Nomeadas (NER) e Similaridade Semântica, comprovando a eficácia da transferência de aprendizado para o português.  
Em uma aplicação para o setor bancário brasileiro, Angelo et al. (2024) compararam modelos baseados em BERT e LLMs para a classificação de intenções e roteamento de mensagens. O estudo concluiu que modelos baseados em embeddings (como o BERT) atingiram precisão superior (\~98,9%) comparados à classificação zero-shot de LLMs (\~53,7%), além de reduzirem os custos operacionais em 96%.  
Elvas e Carvalho (2026) avaliaram LMs especializados contra LLMs (como a família Llama) no reconhecimento de entidades em textos clínicos em português. Eles demonstraram que LMs baseados em encoder (ex: MediAlbertina) superam os LLMs generativos em revocação e precisão estruturada, exigindo significativamente menos energia e poder computacional.  
Al-Tuama e Nasrawi (2022) desenvolveram um chatbot focado no entendimento de requisições de e-commerce. O trabalho evidencia que a abordagem tradicional de NER é frequentemente insuficiente para produtos específicos, exigindo arquiteturas customizadas (como BiLSTM) para obter alta precisão na extração de intenções e entidades.  
Kumar et al. (2020) exploraram a sensibilidade do BERT a textos com ruído, especificamente erros de ortografia e de digitação. Eles demonstraram que o desempenho do BERT em tarefas fundamentais de PLN, como análise de sentimentos e similaridade textual, sofre uma degradação significativa na presença de ruído. O estudo identificou que a principal causa dessa queda é o tokenizador WordPiece, que acaba quebrando palavras com erros ortográficos em subpalavras cujos significados podem ser muito diferentes da palavra original, alterando completamente o sentido da frase.

## 

## 

## **Tabela Comparativa**

| Trabalho | Tarefa(s) | Dados / Domínio | Técnica Principal | Resultados | Diferenças para o nosso Projeto |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Souza et al. (2020) | NER e Similaridade | brWaC, HAREM / Geral | BERTimbau (Fine-tuning) | Estado da arte no PT-BR. | Não aborda textos altamente informais e ruidosos das redes sociais. |
| Angelo et al. (2024) | Classificação de Intenção e Roteamento | Dúvidas bancárias e Spam | Embeddings (BERT) e LLMs | Modelos BERT superou LLMs zero-shot com 5-10% de acurácia  | Foco em setor bancário formal; o projeto foca em pequenos negócios informais via redes sociais. |
| Elvas & Carvalho (2026) | NER Clínico Estruturado | Relatórios médicos (PT-PT) | SLMs, LLMs vs Encoder (MediAlbertina) | LMs Encoders tem 20% a mais que LLMs em precisão e custo energético reduzido em 571% | Foco no domínio formal/médico, o projeto foca no domínio de serviços locais informais. |
| Al-Tuama (2022) | Classificação de Intenção e NER | Produtos de escritório (Amazon) | BiLSTM, seq2seq | 94,75% de acurácia em intenção. | Usa BiLSTM para e-commerce em inglês; este projeto usa BERTimbau para PT-BR ruidoso. |
| Kumar et al., (2020) | Análise de sentimentos e similaridade textual | IMDB, SST-2, STS-B (ruído sintético) | BERT e avaliação de tokenizadores | Desempenho do BERT com 20% de erro no dataset reduz a acurácia em 60% | Foca em ruído no inglês; lidaremos com ruído no português |

## 

## **Comparação Crítica e Lacuna de Pesquisa**

Embora os trabalhos supracitados demonstram a eficácia de modelos baseados em BERT e LLMs em diversos domínios de processamento de linguagem natural, a grande maioria concentra-se em textos formais ou domínios estruturados e regulados, como o bancário, médico e de e-commerce institucional. As soluções propostas lidam com dados que não refletem a alta informalidade típica da comunicação em redes sociais.  
O presente projeto preenche uma lacuna importante ao focar especificamente no nicho de pequenos negócios e prestadores de serviços locais, onde a comunicação via mensagens diretas é dominada por gírias, contrações gramaticais e ausência de pontuação. Diferente das abordagens anteriores, este trabalho adapta o modelo pré-treinado BERTimbau a dados sinteticamente gerados e intencionalmente corrompidos para mimetizar interações de redes sociais autênticas. Além disso, introduzimos uma análise direta do *trade-off* financeiro e operacional, demonstrando analiticamente como um modelo ajustado localmente serve como uma etapa de triagem viável e econômica, mitigando a dependência e os altos custos de inferência associados ao consumo contínuo de tokens em APIs comerciais de LLMs.