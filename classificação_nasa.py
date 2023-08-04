# -*- coding: utf-8 -*-

"""É necessário a instalação abaixo para realizar o balanceamento."""

pip install scikit-plot

"""Importação dos pacotes necessários."""

# importar os pacotes necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
# configurar o estilo dos gráficos com o Seaborn
sns.set_style('dark')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from pylab import plot, show

"""Fazemos a leitura da base de dados e a limitamos a 3000 instâncias, já que ela possui 90000 instâncias."""

nasa = nasa.head(3000)     # LIMITANDO A BASE DE DADOS

nasa

"""Selecionamos as features necessárias para o modelo e excluímos as desnecessárias. Após isso, colocamos os valores das features na variável x_nasa e os valores da saída desejada na variável y_nasa."""

feature_names = ['est_diameter_min',	'est_diameter_max',	'relative_velocity',	'miss_distance',	'absolute_magnitude']   # SELECIONA AS FEATURES QUE SÃO ÚTEIS PARA O MODELO

nasa.dropna(subset=feature_names, inplace=True)      # EXCLUI AS COLUNAS DESNECESSÁRIAS

x_nasa = nasa[feature_names]                         # PEGA OS VALORES DAS FEATURES QUE SÃO IMPORTANTES PARA O MODELO
y_nasa = nasa['hazardous']                           # PEGA A SAÍDA DESEJADA

x_nasa
y_nasa

# USADO PARA FAZER A DIVISÃO DOS DADOS DE TREINO E DE TESTE
x_nasa_train, x_nasa_test, y_nasa_train, y_nasa_test = train_test_split(x_nasa, y_nasa, train_size = 0.75, test_size = 0.25, random_state=123)

"""Apresentamos o tamanho das variáveis de treino e de teste."""

print('Tamanho de x_nasa_train: ', x_nasa_train.shape) # QUANTIDADE DE DADOS DO X NASA TRAIN
print('Tamanho de x_nasa_test: ', x_nasa_test.shape)   # QUANTIDADE DE DADOS DO X NASA TESTE
print('Tamanho de y_nasa_train: ', y_nasa_train.shape) # QUANTIDADE DE DADOS DO Y NASA TRAIN
print('Tamanho de y_nasa_test: ',  y_nasa_test.shape)  # QUANTIDADE DE DADOS DO Y NASA TESTE



"""Salva e treina o modelo da árvore de decisão. Após isso, calculamos a acuracia para o conjunto de dados de teste e de treino. E então, apresentamos a matriz de confusão e a árvore."""

from sklearn.tree import DecisionTreeClassifier # IMPORTA O CLASSIFICADOR DA ÁRVORE DE DECISÃO

model = DecisionTreeClassifier(max_depth=2, min_samples_split=20, random_state=123) # SALVA O CLASSIFICADOR

model.fit(x_nasa_train, y_nasa_train) # TREINA O MODELO

from sklearn.model_selection import cross_val_score

acuracia_train = 100 * cross_val_score(model, x_nasa_train, y_nasa_train).mean() # CALCULA A ACURÁCIA PARA O TREINO
print(acuracia_train)

acuracia_test = 100 * cross_val_score(model, x_nasa_test, y_nasa_test).mean() # CALCULA A ACURACIA PARA O TESTE
print(acuracia_test)

import matplotlib as mpl # IMPORTA O MATPLOTLIB
mpl.rcParams['figure.dpi'] = 100 # DEFINE O TAMANHO DA IMAGEM PARA 100
import matplotlib.pyplot as plt # IMPORTA O PYPLOY
from sklearn.tree import plot_tree # IMPORTA PARA DESENHAR AS ÁRVORES
from sklearn.metrics import ConfusionMatrixDisplay # IMPORTE PARA A MATRIZ DE CONFUSÃO
from sklearn.metrics import classification_report # IMPORTE PARA O RELATÓRIO DE CLASSIFICAÇÃO

y_nasa_pred = model.predict(x_nasa_test) # SALVA A PREVISÃO DO TESTE

ConfusionMatrixDisplay.from_estimator(model, x_nasa_test, y_nasa_test) # PLOTA A MATRIZ DE CONFUSÃO
print(classification_report(y_nasa_test, y_nasa_pred)) # PLOTA O RELATÓRIO

plt.figure() # DESENHA EM FORMATO DE FIGURA
plot_tree(model, filled=False) # DESENHA EM FORMATO DE ÁRVORE
plt.show() # DESENHA A ÁRVORE DE DECISÃO

"""Salvamos e treinamos o modelo da floresta aleatória, calculando a acurácia para os dados de treino e de teste. Depois, apresentamos a matriz de confusão e a árvore."""

from sklearn.ensemble import RandomForestClassifier # IMPORTA A FLORESTA ALEATÓRIA

model = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=123, n_jobs=-1) # SALVA O CLASSIFICADOR

model.fit(x_nasa_train, y_nasa_train) # TREINA O CLASSIFICADOR

acuracia_train = 100 * cross_val_score(model, x_nasa_train, y_nasa_train).mean() # CALCULA A ACURACIA PARA O TREINO
print(acuracia_train)

print('\n')

acuracia_test = 100 * cross_val_score(model, x_nasa_test, y_nasa_test).mean() # CALCULA A ACURACIA PARA O TESTE
print(acuracia_test)

y_nasa_pred = model.predict(x_nasa_test) # SALVA A PREVISÃO

ConfusionMatrixDisplay.from_estimator(model, x_nasa_test, y_nasa_test) # PLOTA A MATRIZ DE CONFUSÃO
print(classification_report(y_nasa_test, y_nasa_pred)) # PLOTA O RELATÓRIO
mpl.rcParams['figure.dpi'] = 1000 #AUMENTA O TAMANHO DA IMAGEM DA FLORESTA

plt.figure() # PLOTA EM FORMA DE FIGURA
plot_tree(model.estimators_[499], filled=True) # PLOTA EM FORMA DE ÁRVORE
plt.show() # PLOTA A FLORESTA

"""Salvamos e treinamos o modelo do XGBoost, calculando a acurácia para o conjunto dos dados de treino e de teste. Após isso, apresentamos a matriz de confusão e a árvore."""

import xgboost as xgb # IMPORTA O XGBOOST

model = xgb.XGBClassifier(n_estimators=500, max_depth=5, random_state=123, n_jobs=-1) # SALVA O CLASSIFICADOR

model.fit(x_nasa_train, y_nasa_train) # TREINA O CLASSIFICADOR

acuracia_train = 100 * cross_val_score(model, x_nasa_train, y_nasa_train).mean() # CALCULA A ACURACIA PARA O TREINO
print(acuracia_train)

print('\n')

acuracia_test = 100 * cross_val_score(model, x_nasa_test, y_nasa_test).mean() # CALCULA A ACURACIA PARA O TESTE
print(acuracia_test)

mpl.rcParams['figure.dpi'] = 100 # MODIFICA O TAMANHO DA IMAGEM

y_nasa_pred = model.predict(x_nasa_test) # SALVA A PREVISÃO

ConfusionMatrixDisplay.from_estimator(model, x_nasa_test, y_nasa_test) # PLOTA A MATRIZ DE CONFUSÃO
print(classification_report(y_nasa_test, y_nasa_pred)) # PLOTA O RELATÓRIO
mpl.rcParams['figure.dpi'] = 1000 # AUMENTA O TAMANHO DA IMAGEM PARA O XGBOOST

plt.figure() # PLOTA EM FORMA DE FIGURA
xgb.plot_tree(model) # PLOTA EM FORMA DE ARVORE
plt.show() # PLOTA O XGBOOST

"""Fazemos o balanceamento de dados, usando a técnica de UNDER-SAMPLING, na qual exclui algumas instâncias para balancear os dados. Feito isso, apresentamos a nova distruibuição de classes e matriz de confusão."""

# USA A TECNICA UNDER-SAMPLING
rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(x_nasa_train, y_nasa_train)

mpl.rcParams['figure.dpi'] = 100
# MOSTRA O BALANCEAMENTO DAS CLASSES
print(pd.Series(y_res).value_counts())

# PLOTA A NOVA DISTRIBUIÇÃO DAS CLASSES
sns.countplot(y_res)

# INSTANCIA E TREINA O MODELO DE ARVORE DE DECISAO
model_res = DecisionTreeClassifier(max_depth=2, min_samples_split=20, random_state=123) # SALVA O CLASSIFICADOR
model_res.fit(X_res, y_res)

# FAZ AS PREVISÕES EM CIMA DO TESTE
y_pred_res = model_res.predict(x_nasa_test)
y_proba_res = model_res.predict_proba(x_nasa_test)

mpl.rcParams['figure.dpi'] = 100 # MODIFICA O TAMANHO DA IMAGEM

# PLOTA A MATRIZ DE CONFUSÃO
skplt.metrics.plot_confusion_matrix(y_nasa_test, y_pred_res, normalize=True)

# IMPRIME O RELATORIO DE CLASSIFICAÇÃO
print("\n\nRelatório de Classificação:\n", classification_report(y_nasa_test, y_pred_res, digits=4))

# IMPRIME A ACURACIA DO MODELO
print("Acurácia: {:.4f}\n".format(accuracy_score(y_nasa_test, y_pred_res)))

"""Fazemos vários testes, modificando o tamanho do conjunto de treino, e mantendo o conjunto de teste. O tamanho do conjunto de teste mantém em 0.25, e o conjunto de treino varia de 0.10 a 0.75. Fazemos os testes para os três modelos e mostramos abaixo, sendo árvore de decisão, floresta aleatória e XGBoost, respectivamente."""

arvore_decisao = pd.read_csv("/content/drive/MyDrive/Semestre Atual /Inteligência Artificial/NASA/Comparação Modelo - Arvore de Decisão - Classificação.csv")

arvore_decisao

floresta_aleatoria = pd.read_csv("/content/drive/MyDrive/Semestre Atual /Inteligência Artificial/NASA/Comparação Modelo - Floresta Aleatória - Classificação.csv")

floresta_aleatoria

xgboost = pd.read_csv("/content/drive/MyDrive/Semestre Atual /Inteligência Artificial/NASA/Comparação Modelo - XGBoost - Classificação.csv")

xgboost