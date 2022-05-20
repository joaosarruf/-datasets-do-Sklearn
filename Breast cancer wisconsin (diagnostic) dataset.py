# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:15:16 2022

@author: JoaoSarruf
"""

# Descrição do Programa: Esse programa detecta cancer de mama baseado em dados

#Importando as bibliotecas


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 


# Carregando o dataframe do sklearn 

df = pd.read_csv(r"C:\Users\jpsar\OneDrive\Área de Trabalho\LPAA\data.csv")
df.head(7)
#Contando o numero de colunas e fileiras do dataset
df.shape

#Nosso dataframe tem o tamanho de (569, 33) , ou seja 33 colunas.
#Contando os vazios (NaN, NAN, na) valores em cada coluna
df.isna().sum()

#REmovendo a coluna ‘Unnamed: 32’ do dataset original ja que nao acrescenta nenhum valor.
df = df.dropna(axis=1)

#Contando o novo valor de colunas e fileiras
df.shape

#Agora nosso dataframe tem o tamanho de (569, 32) , ou seja 32 colunas.

#Pegar o total de numero de pacientes com cancer Maligno (M) e Benigno (B).
df['diagnosis'].value_counts()

# Cancer maligno(M): 212 e benigno(B): 357

#Podemos vizualizar esses valores
sns.countplot(df['diagnosis'],label="Count")

# Agora vamos olhar nossos datos e confirmar que temos todos como numeros e transformar as que forem necessarias 
df.dtypes

#Se pode observar que todas estao como numeros exceto a coluna diagnosis que esta como objeto
# A coluna diagnosis tem como valores 'M' e 'B' aos quais mudaremos para 1 e 0  respectivamente

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))



#Criando um pairplot
cols = ['diagnosis',
        'radius_mean', 
        'texture_mean', 
        'perimeter_mean', 
        'area_mean', 
        'smoothness_mean', 
        'compactness_mean', 
        'concavity_mean',
        'concave points_mean', 
        'symmetry_mean', 
        'fractal_dimension_mean']
sns.pairplot(data=df[cols], hue='diagnosis', palette='rocket')
#Printando as primeiras 5 colunbas do novo dataset 
df.head(5)


#Agora vamos pegar a correlação das colunas 
df.corr()

#Em seguida vamos vizualizar essa correlação com um mapa de calor
plt.figure(figsize=(20,20))  
sns.heatmap(df.corr(), annot=True, fmt='.0%')

#agora vamos dividir o dataset em x e y sendo x o data set independente e y como o data set dependente

X = df.iloc[:, 2:31].values 
Y = df.iloc[:, 1].values 

#Dividindo os dados novamente, mas desta vez serao dividindo em 75% para trainamento e 25% para teste.

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Escalando os dados para que fique tudo no mesmo nivel de magnitude
#O que significa que a "Feature"/dados independentes  virao com um tamanho definido.

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Criando uma função que mantenha mais de um modelo(KNN e Random Forest Classifier) para fazer a classificação.
#Esses são os modelos que detectarão se um paciente tem cancer ou nao. Com essa função eu tambem irei printar a  
#precisão de cada modelo.





def models(X_train,Y_train):
    #Usando KNeighborsClassifier 
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)
  
  #Usando RandomForestClassifier 
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  
  #Printando a precisão dos modelos no treinamento.
  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  print('[2]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
  return knn, forest

#Criando o modelo que contenha ambos modelos, e olhando para a o score de precisão no treinamento 

model = models(X_train,Y_train)

# Mostrando a matrix de confusão e a precisão dos modelos no teste do data.
# A Matrix de confusão nos diz quantos pacientes em cada modelos foram diagnosticados de maneira errada(Falso negativo e Falso positivo)
# E tambem o numero de diagnosticos corretos, positivos positivos e verdadeiros negativos

from sklearn.metrics import confusion_matrix
for i in range(len(model)):
  cm = confusion_matrix(Y_test, model[i].predict(X_test))
  
  TN = cm[0][0]
  TP = cm[1][1]
  FN = cm[1][0]
  FP = cm[0][1]
  
  print(cm)
  print('Model[{}] Testing Accuracy = "{}!"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
  print()


# Mostrando outra maneira de consguir a precisão e outras metricas da classificação 

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
  print('Model ',i)
  #Confere a precisão, recall e  f1-score
  print( classification_report(Y_test, model[i].predict(X_test)) )
  #Outra maneira de conseguir a precisão do modelo nos testes
  print( accuracy_score(Y_test, model[i].predict(X_test)))
  print()#Print a new line


# Pode-se observar que o modelo que conseguiu o melhor resultado foi o Random Forest Classifier
# Conseguindo uma precisão de 96,5% 


# Agora vamos tentar melhorar a precisão do nosso modelo de KNN


from sklearn.neighbors import KNeighborsClassifier
clf =  KNeighborsClassifier(n_neighbors = 5)

clf.get_params()
#Esses sao os parametros do modelo:
"""
Out[38]: 
{'algorithm': 'auto',
 'leaf_size': 30,
 'metric': 'minkowski',
 'metric_params': None,
 'n_jobs': None,
 'n_neighbors': 5,
 'p': 2,
 'weights': 'uniform'}

"""

# Definindo os parametros nos vamos modifica-los

n_neighbors = [1,5,10,20,50,100]

test_accuracy = {}
train_accuracy = {}

for n in n_neighbors:
    clf = KNeighborsClassifier(n_neighbors = n)
    clf.fit(X_train, Y_train)
    test_acc = clf.score(X_test, Y_test)
    train_acc = clf.score(X_train, Y_train)
    test_accuracy[n] = test_acc
    train_accuracy[n] = train_acc

for k, v in test_accuracy.items():
    print("test accuracy for {} n_neighbors is {} %".format(k, round(v*100, 3)))
    
for k, v in train_accuracy.items():
    print("training accuracy for {} n_neighbors is {} %".format(k, round(v*100, 3)))
    
#Em testes mantemos a alta em 95.8%




# Agora vamos tentar melhorar a precisão do nosso modelo de Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 10)

clf.get_params()
#Esses sao os parametros do modelo:
"""
Out[46]: 
{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 10,
 'n_jobs': None,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}
"""

# Definindo os parametros nos vamos modifica-los

n_estimators = [1,5,10,20,50,100]

test_accuracy = {}
train_accuracy = {}

for n in n_estimators:
    clf = RandomForestClassifier(n_estimators = n)
    clf.fit(X_train, Y_train)
    test_acc = clf.score(X_test, Y_test)
    train_acc = clf.score(X_train, Y_train)
    test_accuracy[n] = test_acc
    train_accuracy[n] = train_acc

for k, v in test_accuracy.items():
    print("test accuracy for {} n_estimators is {} %".format(k, round(v*100, 3)))
    
for k, v in train_accuracy.items():
    print("training accuracy for {} n_estimators is {} %".format(k, round(v*100, 3)))



#Em testes conseguimos subir de 96,5% para 97.902











