# ==============================================================================
# ATIVIDADE 2: CHATBOT VERSÃO 2 (DECISION TREE)
# ==============================================================================

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# 1. Carregar dataset
df = pd.read_csv('dataset_moveis_100.csv')


# 2. Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    df['texto'],
    df['intencao'],
    test_size=0.30,
    random_state=42,
    stratify=df['intencao']
)


# 3. Criar a Pipeline
pipeline_arvore = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
    ('classifier', DecisionTreeClassifier(random_state=42))
])


# 4. Treinar o modelo
pipeline_arvore.fit(X_train, y_train)


# 5. Fazer as previsões
y_pred = pipeline_arvore.predict(X_test)


# 6. Mostrar a matriz de confusão
print("\n=== MATRIZ DE CONFUSAO ===")
print(confusion_matrix(y_test, y_pred))


# 7. Mostrar o relatório de classificação
print("\n=== RELATORIO DE CLASSIFICACAO ===")
print(classification_report(y_test, y_pred))


# ==============================================================================

LIMIAR_CONFIANCA = 0.50

print("\n=== INICIANDO BATERIA DE TESTES (8 INPUTS OBRIGATORIOS) ===")


# Palavras conhecidas pelo modelo
vocabulario = set(
    pipeline_arvore.named_steps['vectorizer'].get_feature_names_out()
)


for i in range(1, 9):

    print(f"\n[Teste {i}/8]")

    frase = input("Digite a frase do cliente: ").strip()


    # Verificar se a frase possui palavras que o modelo nao conhece
    palavras = frase.lower().split()

    palavra_desconhecida = False

    for palavra in palavras:

        palavra_limpa = palavra.strip(".,!?;:")

        if palavra_limpa not in vocabulario:

            palavra_desconhecida = True
            break


    # Palavra desconhecida = atendimento humano
    if palavra_desconhecida:

        print(
            "Desculpe, nao entendi sua solicitacao. "
            "Encaminhando voce para um atendente humano..."
        )

        continue


    # Previsao
    probs = pipeline_arvore.predict_proba([frase])[0]

    maior_prob = np.max(probs)

    intencao = pipeline_arvore.predict([frase])[0]


    # Regra de confianca
    if maior_prob >= LIMIAR_CONFIANCA:

        print(
            f"Intencao: {intencao} | "
            f"Confianca: {maior_prob * 100:.1f}%"
        )

    else:

        print(
            "Desculpe, nao entendi sua solicitacao. "
            "Encaminhando voce para um atendente humano..."
        )




# O que saiu no terminal e o que foi escrito:
=== MATRIZ DE CONFUSAO ===
[[4 0 0 0 2]
 [1 2 0 3 0]
 [0 0 6 0 0]
 [0 0 0 5 1]
 [0 0 0 0 6]]

=== RELATORIO DE CLASSIFICACAO ===
                    precision    recall  f1-score   support

logistica_entregas       0.80      0.67      0.73         6
       reclamacoes       1.00      0.33      0.50         6
           suporte       1.00      1.00      1.00         6
 trocas_devolucoes       0.62      0.83      0.71         6
            vendas       0.67      1.00      0.80         6

          accuracy                           0.77        30
         macro avg       0.82      0.77      0.75        30
      weighted avg       0.82      0.77      0.75        30


=== INICIANDO BATERIA DE TESTES (8 INPUTS OBRIGATORIOS) ===

[Teste 1/8]
Digite a frase do cliente: Rastreio do pedido
Intencao: trocas_devolucoes | Confianca: 100.0%

[Teste 2/8]
Digite a frase do cliente: Atraso na entrega
Intencao: trocas_devolucoes | Confianca: 100.0%

[Teste 3/8]
Digite a frase do cliente: Produto com defeito
Intencao: trocas_devolucoes | Confianca: 100.0%

[Teste 4/8]
Digite a frase do cliente: Montagem de armario
Intencao: vendas | Confianca: 100.0%

[Teste 5/8]
Digite a frase do cliente: Preço do colchão 
Desculpe, nao entendi sua solicitacao. Encaminhando voce para um atendente humano...

[Teste 6/8]
Digite a frase do cliente: palmeiras não tem mundial
Desculpe, nao entendi sua solicitacao. Encaminhando voce para um atendente humano...

[Teste 7/8]
Digite a frase do cliente: quero ver hack de televisão
Desculpe, nao entendi sua solicitacao. Encaminhando voce para um atendente humano...

[Teste 8/8]
Digite a frase do cliente: Entrega chega quando
Intencao: trocas_devolucoes | Confianca: 100.0%


