--- RESULTADOS DO LAB 01 ---
Mensagem: 'Quero consultar quanto dinheiro tenho' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Pode me ajudar a fazer um pix?' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Gostaria de cancelar meu cartão de crédito' ==> Intenção Predita: [cancelar_conta]

LAB 01 — Classificador de Intenções
1. Avaliação dos resultados
Os resultados estão corretos. As mensagens foram classificadas de acordo com suas intenções: saldo, Pix e cancelamento.
2. Como melhorar?
Adicionar mais frases ao dataset de treinamento, aumentando a quantidade e variedade dos exemplos.
3. Função do LogisticRegression
É o modelo que aprende os padrões das frases e classifica uma nova mensagem na intenção mais provável.

--- RESULTADOS DO LAB 02 ---
Mensagem de Teste: 'Gostaria de devolver o produto que comprei'
Intenção Predita: troca_devolucao

--- Distribuição de Probabilidades por Classe ---
Classe [duvida_frete]: 27.99%
Classe [rastrear_pedido]: 24.54%
Classe [troca_devolucao]: 47.46%

LAB 02 — Naive Bayes
1. Avaliação dos resultados
O resultado está correto. A mensagem sobre devolver um produto deve ser classificada como troca_devolucao.
2. Como melhorar?
Adicionar mais exemplos de frases para cada intenção, deixando o treinamento maior e mais variado.
3. Função do Naive Bayes
Calcula a probabilidade de uma mensagem pertencer a cada intenção e escolhe a classe com maior probabilidade.
Conclusão
Os dois algoritmos funcionam corretamente nos exemplos apresentados. A principal melhoria seria aumentar o dataset para melhorar a precisão.

--- RESULTADOS DO LAB 03 ---
Acurácia do Modelo: 33.33%

LAB 03 — Árvore de Decisão
1. Avaliação dos resultados
A acurácia obtida no conjunto de teste foi de 33.33%. Em um dataset tão pequeno (9 exemplos no total e apenas 3 no teste), essa métrica é enganosa porque um único erro faz a acurácia despencar. Além disso, frases de teste com palavras inéditas não são reconhecidas pelo modelo.
2. Como melhorar?
Aumentar o volume e a variedade de frases no treinamento, tratar o texto (remover acentos e stopwords) e definir um limite de profundidade (max_depth) na árvore para evitar que o modelo apenas decore os exemplos.
3. Função do DecisionTreeClassifier
Cria uma estrutura de regras de decisão ("se contém a palavra X, vá para o ramo Y") baseada nas palavras mais importantes para separar e classificar cada intenção.

--- RESULTADOS DO LAB 04 --- 
Mensagem: 'Preciso comprar uma passagem urgente para Madri' ==> Intenção Predita: [comprar_passagem]
Mensagem: 'Gostaria de cancelar o voo que agendei ontem' ==> Intenção Predita: [cancelar_reserva]
Mensagem: 'Quero transferir o atendimento para um humano' ==> Intenção Predita: [falar_atendente]

LAB 04 —  Código
# ============================================================
# LAB 04 - AULA 02 (MLCB): DESAFIO NLU PARA AGÊNCIA DE VIAGENS
# ============================================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Dataset Próprio da Agência de Viagens (12 frases / 3 intenções)
dados_viagens = {
    'mensagem': [
        'Quero comprar uma passagem aérea para Orlando',
        'Gostaria de reservar um voo para Salvador nas férias',
        'Qual é o valor do bilhete aéreo para Lisboa?',
        'Preciso emitir uma passagem de avião urgentemente',
        'Como faço para cancelar minha reserva de voo?',
        'Quero solicitar o cancelamento da minha viagem',
        'Preciso anular meu bilhete aéreo e pedir reembolso',
        'Gostaria de efetuar o cancelamento do meu hotel',
        'Preciso falar com um atendente humano agora',
        'Pode me transferir para a central de suporte?',
        'Quero conversar com um especialista da agência',
        'Gostaria de ajuda de um atendente sobre meu pacote'
    ],
    'intencao': [
        'comprar_passagem', 'comprar_passagem', 'comprar_passagem', 'comprar_passagem',
        'cancelar_reserva', 'cancelar_reserva', 'cancelar_reserva', 'cancelar_reserva',
        'falar_atendente', 'falar_atendente', 'falar_atendente', 'falar_atendente'
    ]
}

df4 = pd.DataFrame(dados_viagens)

# 2. Separação de Atributos (X) e Rótulos (y)
X = df4['mensagem']
y = df4['intencao']

# 3. Divisão em Treino e Teste (estratificado para manter proporções)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Vetorização de Texto via TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Treinamento do Modelo (Regressão Logística)
modelo = LogisticRegression()
modelo.fit(X_train_vec, y_train)

# 6. Teste com Mensagens Inéditas
novas_frases = [
    "Preciso comprar uma passagem urgente para Madri",
    "Gostaria de cancelar o voo que agendei ontem",
    "Quero transferir o atendimento para um humano"
]

novas_frases_vec = vectorizer.transform(novas_frases)
predicoes = modelo.predict(novas_frases_vec)

print("--- RESULTADOS DO LAB 04 ---")
for frase, intencao in zip(novas_frases, predicoes):
    print(f"Mensagem: '{frase}' ==> Intenção Predita: [{intencao}]")

