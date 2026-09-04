AC-2 Parte 1 - resultados_aula05

Integrantes

Eu, João e Manuel.

A IA foi usada como apoio para entender o código, corrigir erros e conferir os resultados. Nós executamos e acompanhamos a atividade no Google Colab.

Exercício 1

A função faz limpeza do texto, remove stop words e realiza a lematização.

Resultado:

Frase Original: Gostaria de saber se vocês estão DEVOLVENDO os valores das mesas compradas!!!
Frase Limpa & Lemmatizada: gostar saber devolver valor meso comprada

Exercício 2

Foi utilizado Mean Pooling para calcular a média dos vetores.

Resultado:

(32, 50)

Exercício 3

Foi utilizada Regressão Logística com limite de 50% para o fallback.

Resultados:

Frase: 'Quero saber o valor do frete do sofá' | Resultado: FALLBACK_HUMANO | Confiança: 35.52%
Frase: 'Gostaria de ver receitas de bolo de cenoura' | Resultado: vendas_orcamento | Confiança: 69.10%

Exercício 4

Foi utilizado KNN com K=3.

Acurácia - Regressão Logística (Linear): 93.75%
Acurácia - KNN (Distância K=3): 56.25%

Reflexão

O KNN utiliza a distância entre os vetores e procura os vizinhos mais próximos. Porém, nesta base, a Regressão Logística teve melhor resultado.
