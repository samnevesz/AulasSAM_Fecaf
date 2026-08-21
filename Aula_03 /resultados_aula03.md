--- RESULTADOS DO LAB 01 (AULA 03) ---
Mensagem: 'Preciso urgente da segunda via da fatura'
Intenção Predita: [segunda_via]
Vocabulário Filtrado (sem stopwords): ['2a', '2a via', 'aberto', 'acordo', 'acordo pagar', 'alterar', 'alterar endereço', 'app', 'atrasada', 'atualizo', 'atualizo dados', 'boleto', 'cadastramento', 'dados', 'dados residenciais', 'débito', 'débito aberto', 'dívida', 'emitir', 'emitir segunda', 'endereço', 'endereço cadastramento', 'fatura', 'fatura atrasada', 'fazer', 'fazer um', 'gostaria', 'gostaria alterar', 'negociar', 'negociar pagamento', 'no', 'no app', 'onde', 'onde atualizo', 'pagamento', 'pagamento dívida', 'pagar', 'pagar débito', 'posso', 'posso emitir', 'residenciais', 'residenciais no', 'segunda', 'segunda via', 'um', 'um acordo', 'via', 'via boleto', 'via fatura']

1. Impacto da Remoção de Stopwords
A remoção reduz drasticamente a dimensão da matriz de termos, eliminando ruído e palavras com alta frequência que não possuem valor semântico discriminativo (como preposições e artigos). Isso diminui a complexidade computacional do modelo e foca os pesos do TF-IDF apenas em palavras com real valor preditivo.

2. Significado de ngram_range=(1, 2)
A instrução extrai tanto unigramas (palavras individuais) quanto bigramas (pares de palavras consecutivas). No script enviado, o vetorizador captura termos isolados como segunda e via, além da expressão composta segunda via. Isso garante a preservação do contexto local e de termos compostos cruciais para a intenção.

3. Prevenção de Classificações Incorretas
Palavras genéricas (ex: como, preciso, minha) aparecem em mensagens de qualquer intenção. Se não forem removidas, o TF-IDF atribui peso a termos semântica irrelevantes. Sem elas, o modelo aloca importância estritamente às palavras-chave que diferenciam as classes (ex: boleto, dívida, endereço), evitando sobreajuste em ruídos gramaticais.

--- RESULTADOS DO LAB 02 (AULA 03) ---

--- Relatório de Classificação ---
                     precision    recall  f1-score   support

horario_atendimento       0.50      1.00      0.67         1
        localizacao       0.00      0.00      0.00         1
    troca_devolucao       0.00      0.00      0.00         1

           accuracy                           0.33         3
          macro avg       0.17      0.33      0.22         3
       weighted avg       0.17      0.33      0.22         3

--- Matriz de Confusão ---
[[1 0 0]
 [1 0 0]
 [0 1 0]]

1. O que representam Precision, Recall e F1-Score?
Precision (Precisão): Da quantidade de vezes que o modelo previu uma classe, quantas ele acertou. Mede a confiabilidade da previsão.
Recall (Revocação): De todos os exemplos reais que existiam de uma classe, quantos o modelo conseguiu encontrar. Mede a capacidade de detecção.
F1-Score: O equilíbrio entre Precision e Recall (média harmônica). Resume o desempenho geral da classe em um único número.

2. Como interpretar a diagonal principal da Matriz de Confusão?
A diagonal principal mostra os acertos do modelo (Verdadeiros Positivos).
Cada número nela indica quantas mensagens de uma classe foram classificadas corretamente na própria classe.
Quanto maiores os valores na diagonal principal (e menores fora dela), melhor é o desempenho.

3. Por que a acurácia isolada pode ser enganosa em classes desbalanceadas?
Porque a acurácia calcula apenas o total de acertos sobre o total geral, ignorando o desempenho em classes menores.
Exemplo: Se 95% das mensagens forem sobre Localização e 5% sobre Troca, um modelo que chutar sempre "Localização" terá 95% de acurácia, mas errará 100% dos pedidos de Troca.

1.CÓDIGO FEITO
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dados_rh = {
    'mensagem': [
        'Como solicitar minhas ferias?', 'Quero agendar meu periodo de ferias',
        'Onde baixo meu holerite do mes?', 'Preciso do comprovante de rendimentos',
        'Como cadastrar meu atestado medico?', 'Onde envio o atestado de consulta?'
    ],
    'intencao': [
        'solicitar_ferias', 'solicitar_ferias',
        'obter_holerite', 'obter_holerite',
        'enviar_atestado', 'enviar_atestado'
    ]
}

df3 = pd.DataFrame(dados_rh)

# TODO 1: Separe o dataset em X ('mensagem') e y ('intencao')
X = df3['mensagem']
y = df3['intencao']

# TODO 2: Realize o train_test_split com test_size=0.33 e random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# TODO 3: Monte o Pipeline encapsulando o TfidfVectorizer e a LogisticRegression
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=['de', 'o', 'meu', 'minhas'])),
    ('classifier', LogisticRegression())
])

# TODO 4: Treine o pipeline completo com .fit() usando os dados de treino brutos
pipeline.fit(X_train, y_train)

# TODO 5: Faca a predicao nos dados de teste brutos e exiba a acuracia
predicoes = pipeline.predict(X_test)
print(f"Acuracia via Pipeline: {accuracy_score(y_test, predicoes) * 100:.2f}%")

Acuracia via Pipeline: 0.00%

2. Qual é a grande vantagem de utilizar o objeto Pipeline no Scikit-Learn?
A grande vantagem é a automação e a simplificação do código. O Pipeline junta o pré-processamento (transformação dos dados, como o TfidfVectorizer) e o modelo final (como a LogisticRegression) em um único objeto. Assim, você não precisa transformar os dados manualmente em cada etapa; basta chamar .fit() para treinar tudo e .predict() para prever direto sobre o texto bruto.

3. Por que o Pipeline evita que erros de pré-processamento ocorram entre treino e teste?
Ele previne o vazamento de dados (data leakage). Quando você usa o Pipeline, ele aprende o vocabulário e as regras de transformação apenas com os dados de treino (X_train). Na hora de testar (X_test) ou usar em produção, o Pipeline aplica essa mesma regra pronta, sem reajustar nem espiar informações do conjunto de teste.
