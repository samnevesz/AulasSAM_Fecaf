Relatório de Execução — Aula 06 (AC-2 Parte 2)
Atividade: Pipeline NLU Supervisionado para Chatbot Imobiliário

Integrantes: [Samuel Leonardo Oliveira Barbosa, Manoel Rodrigues ADS 2NA]
r
1. Testes da Base Inicial
Para validar o código do professor antes das alterações, rodamos o modelo original (Regressão Logística, Mean Pooling e corte de 50%) com 4 frases no Gradio:

"Quero comprar um apartamento de 3 quartos"

Intenção: comprar_imovel | Confiança: 92.4% | Status: IDENTIFICADO

"Onde pego a segunda via do boleto?"

Intenção: 2via_boleto_contrato | Confiança: 88.1% | Status: IDENTIFICADO

"Vazamento no teto do banheiro"

Intenção: suporte_manutencao | Confiança: 95.0% | Status: IDENTIFICADO

"Vocês vendem terreno na Lua?"

Intenção: comprar_imovel | Confiança: 32.0% | Status: UNCERTAIN (Fallback Acionado)

2. O Que Fizemos nos Laboratórios
LAB 01: Troca para Árvore de Decisão
Substituímos a Regressão Logística pela Árvore de Decisão (DecisionTreeClassifier) importada do sklearn.tree. Mantivemos o resto do código igual. O modelo treinou sem erros e a interface no Gradio continuou classificando as mensagens normalmente.

LAB 02: Mudança para Max Pooling
Na função extrair_sentence_embedding, trocamos o np.mean pelo np.max com o parâmetro axis=0. Com isso, em vez de tirar a média das palavras limpas, o vetor da frase passou a pegar o valor máximo de cada uma das 50 dimensões do GloVe. O sistema continuou funcionando e gerando as respostas da tela.

LAB 03: Ajuste do Fallback para 65%
Alteramos a variável LIMIAR_CONFIANCA de 0.50 para 0.65. Também mexemos na função processar_atendimento_sac para atualizar o texto do status para "UNCERTAIN (Fallback Acionado - Corte 65%)" e forçar o campo da intenção a mostrar "Não Identificado" quando a confiança ficar abaixo de 65%.

LAB 04: Criando a Classe de Cancelamento
Adicionamos 5 frases na lista dados_imobiliaria com a intenção cancelar_contrato (como "Quero cancelar meu contrato de aluguel" e "Solicito o distrato do contrato de compra"). Depois, colocamos a resposta dessa nova classe no dicionário RESPOSTAS_PADRAO. Rodamos o treino de novo e o bot respondeu certo aos testes de rescisão.

3. Como a IA Ajudou Nosso Grupo
Usamos a IA durante a aula como uma ferramenta de apoio para destravar o código e entender melhor o que estávamos fazendo:

Ajudou a resolver o erro de módulo: Quando fomos rodar o código no Colab deu erro no import gensim. A IA nos explicou que essa biblioteca não vem instalada no Colab e passou o comando !pip install gensim pra colocar no topo da célula.

Ajuste da função do NumPy: Ficamos em dúvida sobre como aplicar o Max Pooling sem quebrar o formato da matriz, e a IA nos mostrou exatamente onde trocar o np.mean por np.max(axis=0).

Arrumou um detalhe na tela do Gradio: Percebemos que quando a frase caía no Fallback (ex: a frase da Lua), o sistema mandava a mensagem de erro, mas a caixa "Intenção" ainda mostrava comprar_imovel. A IA deu a ideia de colocar um if/else na função para retornar "Não Identificado" nessa caixa e não confundir quem estivesse testando.

Ganho de tempo: Ajudou a conferir se não estávamos esquecendo nenhuma etapa exigida nos 4 laboratórios antes de montar o relatório.
