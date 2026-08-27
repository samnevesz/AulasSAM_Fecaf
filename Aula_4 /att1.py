# ==============================================================================
# ATIVIDADE 1: CHATBOT VERSÃO 1 (KNN)
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# 1. Carregar dataset do CSV
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
pipeline_knn = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
    ('classifier', KNeighborsClassifier(
        n_neighbors=3,
        metric='cosine'
    ))
])


# 4. Treinar a Pipeline
pipeline_knn.fit(X_train, y_train)


# 5. Fazer as previsões
y_pred = pipeline_knn.predict(X_test)


# 6. Mostrar o relatório de classificação
print("\n=== RELATORIO DE CLASSIFICACAO ===")
print(classification_report(y_test, y_pred))


# 7. Mostrar a matriz de confusão
print("\n=== MATRIZ DE CONFUSAO ===")
print(confusion_matrix(y_test, y_pred))


# ==============================================================================

LIMIAR_CONFIANCA = 0.50

print("\n=== INICIANDO BATERIA DE TESTES (10 INPUTS OBRIGATORIOS) ===")


for i in range(1, 11):

    print(f"\n[Teste {i}/10]")

    # Solicitar a frase do usuario
    frase = input("Digite a frase do cliente: ").strip()


    # Extrair as probabilidades e a classe prevista
    probs = pipeline_knn.predict_proba([frase])[0]

    maior_prob = np.max(probs)

    intencao = pipeline_knn.predict([frase])[0]


    # Regra de decisão
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
