# Detecção de Fraude em Transações de Cartão de Crédito

Desafio de Projeto — Bootcamp Bradesco - GenAI & Dados em parceria com a DIO

## Resumo

Neste projeto, construí um pipeline de aprendizado de máquina para detectar fraudes em transações reais de cartão de crédito, cobrindo desde a exploração dos dados até a explicação das decisões do modelo. Como a fraude é um evento raro (0,17% das transações), a avaliação foi centrada em **recall, precisão e F1 da classe fraude**, e não na acurácia. Comparei regressão logística, Random Forest e XGBoost, ajustei o limiar de decisão em um conjunto de validação separado e interpretei o modelo final com importância de variáveis e SHAP.

O código completo, com todas as saídas salvas, está em [`deteccao_fraude_cartao.ipynb`](deteccao_fraude_cartao.ipynb).

## 1. O problema e o impacto do desbalanceamento

A base utilizada contém 284.807 transações, das quais apenas 492 (0,173%) são fraudes. Esse desbalanceamento altera a forma de avaliar os modelos: um classificador que responde "não é fraude" para todas as transações atinge cerca de 99,83% de acurácia sem detectar nenhuma fraude. A acurácia, portanto, não distingue um modelo útil de um modelo inútil.

Essa limitação apareceu de forma concreta nos meus resultados. No limiar padrão de 0,5, a regressão logística obteve acurácia de 97,41%, inferior à do classificador trivial, e ainda assim tinha o maior recall (91,84%). O motivo é que ela detectou 90 das 98 fraudes do teste ao custo de cerca de 1.470 alarmes falsos (precisão de 5,77%). Por isso, o recall isolado também não basta: é necessário analisá-lo em conjunto com a precisão, o F1 e a curva de precisão e recall.

## 2. Preparação dos dados

- **Variável nova `LogAmount`**, calculada como `log(1 + Amount)`, porque a distribuição do valor é muito assimétrica (as fraudes têm mediana menor, 9,25 contra 22,00, mas média maior, 122,21 contra 88,29). Usei `log1p` para lidar com transações de valor zero.
- **Remoção de `Time` e de `Amount` original.** `Time` é o tempo em segundos desde a primeira transação e, em formato bruto, não generaliza para dados futuros.
- **Padronização** com `StandardScaler`, ajustado apenas no treino para evitar vazamento de dados. Foi aplicada somente à regressão logística, já que os modelos de árvores não dependem de escala.
- **Separação estratificada** (`stratify`) em treino, validação e teste, mantendo a proporção de fraudes (0,17%) em todos os subconjuntos. O teste (20%) ficou reservado para a avaliação final. Do treino, separei 25% como validação, usada exclusivamente para escolher o limiar de decisão.
- **Tratamento do desbalanceamento** por peso das classes (`class_weight="balanced"` e `scale_pos_weight`, este calculado a partir dos dados: 577,29).

## 3. Comparação entre os modelos

Resultados no conjunto de teste (56.962 transações, 98 fraudes), com o limiar padrão de 0,5. Todas as métricas de precisão, recall e F1 se referem à classe fraude.

| Modelo | Precisão | Recall | F1 | AUC-ROC | Average Precision |
|---|---|---|---|---|---|
| Regressão Logística | 0,0577 | 0,9184 | 0,1086 | 0,9705 | 0,7078 |
| Random Forest | 0,9186 | 0,8061 | 0,8587 | 0,9661 | 0,8653 |
| XGBoost | 0,8817 | 0,8367 | 0,8586 | 0,9690 | 0,8779 |

Três observações:

1. A curva ROC (AUC entre 0,966 e 0,971) sugere que os três modelos são equivalentes, e a regressão logística chega a parecer a melhor. A curva de precisão e recall mostra o contrário: a regressão logística é claramente pior (AP de 0,708 contra 0,865 e 0,878). A ROC esconde esse problema porque a taxa de falsos positivos fica muito pequena diante de centenas de milhares de transações normais.
2. Random Forest e XGBoost tiveram desempenho equivalente. Com apenas 98 fraudes no teste, cada fraude a mais ou a menos altera o recall em cerca de 1 ponto percentual, então não considero a diferença entre os dois estatisticamente conclusiva. O XGBoost ficou levemente à frente na precisão média.
3. Ambos os modelos de árvores superaram o baseline linear de forma clara.

## 4. Limiar de decisão e SHAP

### Limiar de decisão

O limiar de 0,5 pressupõe que os dois erros têm o mesmo custo, o que não é o caso aqui. Para escolher o limiar sem usar o conjunto de teste, treinei uma cópia de cada modelo no subconjunto de treino e obtive suas probabilidades no conjunto de validação. Defini duas regras: **(A)** o limiar que maximiza o F1 da fraude e **(B)** o maior limiar que mantém recall de pelo menos 0,90. Os limiares foram então aplicados aos modelos finais no teste.

Resultados no teste com os limiares ajustados:

| Modelo | Regra | Limiar | Precisão | Recall | F1 |
|---|---|---|---|---|---|
| Regressão Logística | A (máx. F1) | 0,99999995 | 0,8316 | 0,8061 | 0,8187 |
| Random Forest | A (máx. F1) | 0,406667 | 0,8632 | 0,8367 | 0,8497 |
| XGBoost | A (máx. F1) | 0,985660 | 0,9740 | 0,7653 | 0,8571 |

**Limiar escolhido:** Para o modelo final, utilizei o **XGBoost** com o limiar **0,985660**, escolhido pela regra A (máximo F1) no conjunto de validação. No teste, esse limiar resultou em precisão de **97,40%**, recall de **76,53%** e F1 de **85,71%**. A regra B, que prioriza recall mínimo de 90%, aumenta o recall, mas reduz fortemente a precisão; por isso, nesta versão do projeto, foi mantida a regra de máximo F1, buscando um compromisso entre os dois tipos de erro.

Observo que, na regressão logística com `class_weight="balanced"`, as probabilidades ficam muito próximas de 1 e o limiar ótimo também. Isso decorre do reponderamento das classes e não indica erro no procedimento.

### O que o SHAP mostrou

Analisei o XGBoost com `TreeExplainer` sobre uma amostra de 2.000 transações do teste. As variáveis de maior impacto foram **V14, V4, V12, V10 e V11**. Valores baixos de V14, V12 e V10 e valores altos de V4 e V11 aumentam a probabilidade prevista de fraude. Como essas variáveis são componentes de PCA, não é possível atribuir-lhes significado de negócio; só consigo descrever a direção do efeito. A variável `LogAmount`, criada por mim, ficou em posição intermediária no ranking, o que indica que o valor da transação pesa menos que os componentes principais.

Na importância de variáveis, Random Forest e XGBoost concordam que V14 é a mais relevante (cerca de 0,55 no XGBoost e 0,16 no Random Forest), mas divergem no restante: o Random Forest distribui a importância entre V10, V4, V12 e V17, enquanto o XGBoost concentra-se mais em V14. Também gerei uma explicação individual (gráfico em cascata) para a transação com maior probabilidade de fraude, mostrando quais variáveis levaram àquela decisão.

## 5. O que modifiquei em relação à aula

Tomando como referência os trechos da aula que utilizei:

- Na aula, o XGBoost foi configurado com `scale_pos_weight=10` fixo e `eval_metric="logloss"`. No meu trabalho, calculei o peso a partir da proporção real das classes no treino (577,29) e usei `eval_metric="aucpr"`, mais adequada a dados desbalanceados.
- Na aula, o SHAP foi aplicado com `shap.Explainer` sobre 100 transações, com gráfico de barras. Utilizei `TreeExplainer` sobre 2.000 transações, com gráfico de resumo (que mostra a direção do efeito, e não só a magnitude) e um gráfico em cascata para um caso individual.
- Escolhi o limiar em um conjunto de validação separado, para que o conjunto de teste permanecesse inédito, e comparei duas regras de escolha (máximo F1 e recall mínimo).
- Incluí a curva de precisão e recall na comparação, por ser mais informativa que a ROC neste problema, e a análise explícita da acurácia do classificador trivial.

- Nesta versão, não apliquei undersampling ou oversampling: o desbalanceamento foi tratado pelo peso das classes. Essas técnicas ficaram como possibilidades de evolução do projeto.

## 6. Limitações

- O conjunto de teste tem apenas 98 fraudes, então as métricas têm incerteza considerável. Uma validação cruzada estratificada daria uma comparação mais robusta.
- Todos os resultados vêm de uma única partição dos dados, e o limiar foi escolhido em uma validação com poucas fraudes, o que o torna uma estimativa ruidosa.
- O limiar foi obtido com modelos treinados em 60% dos dados e aplicado a modelos treinados em 80%; a calibração das probabilidades pode diferir ligeiramente.
- A amostra aleatória usada no SHAP contém poucas fraudes reais; uma análise focada nas fraudes seria mais informativa.
- Por causa do PCA, as variáveis V1 a V28 são anônimas e não permitem traduzir os resultados em regras de negócio.

**Trabalhos futuros:** comparar undersampling e oversampling com o peso de classes, ampliar a busca de hiperparâmetros com `GridSearchCV`, criar variáveis de comportamento ao longo do tempo e testar outros classificadores.

## 7. Como reproduzir

1. Abra `deteccao_fraude_cartao.ipynb` no Google Colab ou no VS Code (com a extensão Jupyter), em um ambiente com internet.
2. Execute a primeira célula para instalar `xgboost`, `shap` e `matplotlib`.
3. Execute as demais células em sequência. A base é carregada diretamente do link público e não está incluída neste repositório. Todas as sementes aleatórias estão fixadas (`random_state=42`).

## Referências

- Base de dados: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
