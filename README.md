# Titanic

from google.colab import drive
drive.mount('/content/drive')

from ast import increment_lineno
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
%matplotlib inline 
%pylab inlin

teste = pd.read_csv('/content/drive/MyDrive/Kaggle/titanic/test.csv')
treino = pd.read_csv('/content/drive/MyDrive/Kaggle/titanic/train.csv')

treino1 = treino
teste1 = teste

modelo = RandomForestClassifier(random_state= 42)

def sexo_bi (sexo):
  if sexo == 'female':
    return 1
  else :
    return 0

treino1['sx'] = treino1['Sex'].map(sexo_bi)

variavel=['sx', 'Age','SibSp','Parch','Fare','Pclass']

x = treino1[variavel]
y = treino1['Survived']

x = x.fillna(-1)

modelo.fit(x,y)

teste1['sx'] = teste1['Sex'].map(sexo_bi)

x_prev = teste1[variavel]
x_prev = x_prev.fillna(-1)

pre = modelo.predict(x_prev)

sub = pd.Series(pre, index = teste1['PassengerId'], name='Survived')

x_treino, x_valid, y_treino, y_valid = train_test_split (x,y, test_size=0.5)

modelo.fit(x_treino,y_treino)

pre = modelo.predict(x_valid)

np.mean(y_valid == pre)

# Validação Cruzada

resultados = []
for rep in range(10):
  print('Rep:',rep)
  kf = KFold(4, shuffle=True, random_state=42)

  for linhas_treino, linhas_valid in kf.split(x):
    print('Treino:',linhas_treino.shape[0])
    print('Valid:',linhas_valid.shape[0])

    x_treino, x_valid = x.iloc[linhas_treino], x.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    modelo = RandomForestClassifier(random_state= 42)
    modelo.fit(x_treino,y_treino)

    pre = modelo.predict(x_valid) 

    acu = np.mean(y_valid == pre) 

    resultados.append(acu)
    print('Acu',acu)
    print()

np.mean(resultados)
