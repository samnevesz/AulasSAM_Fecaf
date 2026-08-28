1. Tabela Comparativa de Métricas
Modelo	Acurácia Geral	F1-Score (Weighted)	Principais Erros na Matriz
KNN (K=3)	100%	100%	Não houve confusão entre as classes
Decision Tree	76,67%	75%	Ocorreram confusões entre vendas, logística, reclamações e trocas/devoluções

2. Análise dos Testes de Entrada
KNN (10 testes): O modelo respondeu bem às frases mais próximas do conteúdo usado no treinamento. Em "ver as mesas", a confiança ficou abaixo de 50% e o sistema acionou o atendimento humano. Por outro lado, "Qual preço da geladeira" foi classificada como logística_entregas com 100%, mesmo sendo um produto que não aparece no dataset.
Decision Tree (8 testes): A árvore teve mais dificuldade para separar algumas intenções. "Rastreio do pedido", "Atraso na entrega" e "Entrega chega quando" foram classificadas como trocas_devolucoes. Já frases sem relação com o dataset, como "palmeiras não tem mundial", foram direcionadas ao atendimento humano pelo fallback.

3. Veredito Final
Melhor modelo para este projeto: KNN (K=3)
Justificativa técnica: O KNN teve desempenho superior nos dados de teste, acertando os 30 casos e alcançando 100% de acurácia e F1-Score Weighted. A Decision Tree acertou 23 dos 30 casos, ficando com 76,67% de acurácia e 75% de F1-Score Weighted. Por apresentar menos erros na classificação, o KNN foi a melhor opção entre os dois modelos.
