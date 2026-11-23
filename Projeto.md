# Classificação de Intenção e Extração de Entidades em Mensagens Informais para Pequenos Negócios.  


## Motivação

Empreendedores de nicho e prestadores de serviços locais, investem considerável tempo diário respondendo a mensagens diretas com dúvidas recorrentes e requisições não estruturadas. 

A automação desse atendimento esbarra na dificuldade dos sistemas genéricos em processar a linguagem altamente informal, gírias e abreviações típicas do contato via redes sociais. Esse problema apresenta um cenário real e prático para o Processamento de Linguagem Natural: converter texto ruidoso em intenções claras (como pedidos de orçamento ou dúvidas de disponibilidade) e entidades estruturadas (como datas, locais e escopo do serviço), aliviando a carga operacional do atendimento humano.  

## Pergunta de pesquisa

Como o desempenho e a eficiência de custo de um modelo pré-treinado adaptado via fine-tuning (BERTimbau) se comparam a baselines clássicos e a Modelos de Linguagem de Grande Escala (LLMs via few-shot prompting) na classificação de intenção e extração de entidades em mensagens informais de serviços locais?

## Dados utilizados

Devido à escassez de dados abertos e específicos para este nicho, o projeto utilizará uma abordagem híbrida de dados. O corpus de treinamento será composto por 944 conversas sintéticas geradas por modelos de linguagem (LLMs) do dataset RichardSakaguchiMS/brazilian-customer-service-conversations, incorporando propositalmente perturbações textuais — como erros ortográficos comuns, abreviações e ausência de pontuação — para simular a comunicação autêntica das redes sociais. Para a validação empírica e avaliação do modelo, será compilado um conjunto de teste com cerca de 50 a 100 mensagens reais de clientes, rigorosamente anonimizadas para garantir a remoção de informações pessoais sensíveis.  

## Resultados esperados

Espera-se desenvolver e avaliar um pipeline de processamento de linguagem capaz de atuar no ecossistema de atendimento local. Os artefatos e conclusões incluirão:  

* A criação de um dataset curado em PT-BR com foco no domínio de serviços informais, que poderá servir para estudos futuros da disciplina.  
* Uma análise comparativa de desempenho (F1-Score, Precisão e Revocação) contrastando o BERTimbau não apenas com os baselines tradicionais, mas também com a capacidade de extração (zero/few-shot) de LLMs comerciais.  
* A demonstração da viabilidade de extrair variáveis do texto livre e transformá-las em chaves-valor estruturadas (JSON), o que permitiria a integração dos resultados extraídos diretamente em consultas rígidas de bancos de dados relacionais.  
* Uma projeção de redução de custos de inferência, demonstrando matematicamente a viabilidade econômica de utilizar o modelo ajustado como uma etapa de triagem de processamento local versus o consumo de tokens exigido por APIs de LLMs em um fluxo de atendimento contínuo.