import numpy as np
import pandas as pd

################################################
############## D A D O S #######################

# Carregar Classes
classes = np.load('Classes.npy', allow_pickle=True)

# Mapear Classes
class_mapping = {'Classe A': 1, 'Classe B': 2, 'Classe C': 3, 'Classe D': 4, 'Classe E': 5}

# Trocar classes nomes por números
for class_name, class_number in class_mapping.items(): 
    classes[classes == class_name] = class_number

# Lista de arquivos ### Dados_4.npy constante (50) e não considerada
file_names = ['Dados_1.npy', 'Dados_2.npy', 'Dados_3.npy']#, 'Dados_5.npy']

# Carrega cada arquivo, verifica o número de colunas, e remove a última se necessário. 
data_list = []
for file_name in file_names:
    data = np.load(file_name)
    if data.shape[1] >= 201:
        data = data[:, :-1]
    # Usando Pandas, remove NaN e troca por mediana
    df = pd.DataFrame(data)
    df_filled = df.fillna(df.median())
    data_structure_filled = df_filled.values
    # Normalização
    normalized_data = (data_structure_filled - np.min(data_structure_filled)) / (np.max(data_structure_filled) - np.min(data_structure_filled))
    data_list.append(data_structure_filled)

# Concatena horizontalmente e adiciona classes no início
data_structure = np.hstack(data_list)
data_structure = np.concatenate((classes,data_structure), axis=1)

data_structure.shape


################################################ 
######### TESTE!!! com dados iniciais ##########

# data_structure = data_structure[:2000, :]


################################################
################ CSV file ######################

# import csv
# filename='stacked_structure.csv'

# with open(filename, 'w', newline='') as csvfile:
    # csvwriter = csv.writer(csvfile)
    # csvwriter.writerows(data_structure)

################################################
################ P C A #########################

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Extrair dados e rótulos (X, y) para PCA e modelos
X = data_structure[:, 1:]  # dados
y = data_structure[:, 0]   # rótulos

# PCA, redução de dimensionalidade para 10 componentes
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Plotagem dos dados
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.colorbar(label='Class')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()

print("################################################")

################################################
################# S V M ########################

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Separar os dados em treinamento e teste (80/20), usando todas as colunas (X) ou só as principais (X_pca)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# OOOOOUUUUU X_pca
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Codificar os rótulos com valores entre 0 e número de classes menos 1
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)

# Define os parâmetros e inicializa o SVM
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['poly', 'sigmoid', 'linear']}
svm_classifier = SVC()
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV aos dados de treinamento
grid_search.fit(X_train, y_train)

# SVM melhores parâmetros
best_params = grid_search.best_params_
print("SVM melhores parâmetros:", best_params)

# Melhor modelo SVM
best_svm_classifier = grid_search.best_estimator_

# Predição dos rótulos de teste
y_pred = best_svm_classifier.predict(X_test)

# Calcula a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print("SVM melhor acurácia com validação cruzada:", accuracy)

# Salva o modelo SVM
joblib.dump(best_svm_classifier, 'svm_model.pkl')

# Reporte de Classificação
print("Reporte de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("################################################")

################################################
################# K N N ########################

from sklearn.neighbors import KNeighborsClassifier

# Número de vizinhos
neighbors = [3, 5, 7, 9, 11]

# Inicialização do KNN
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': neighbors}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV aos dados de treinamento
grid_search.fit(X_train, y_train)
best_knn_model = grid_search.best_estimator_

# Salva o modelo KNN
joblib.dump(best_knn_model, 'knn_model.pkl')

best_n_neighbors = grid_search.best_params_['n_neighbors']
print("KNN melhor número de vizinhos", best_n_neighbors)
best_accuracy = grid_search.best_score_
print("KNN melhor acurácia com validação cruzada:", best_accuracy)

# Reporte de Classificação
print("Reporte de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("################################################")


################################################
############## Random Forest ###################

from sklearn.ensemble import RandomForestClassifier

# Inicialização RF
rf_classifier = RandomForestClassifier()

param_grid = {
    'n_estimators': [50, 100, 200],  # Número de árvores
    'max_depth': [None, 10, 20],      # Profundidade máxima
    'min_samples_split': [2, 5, 10]   # Número mínimo para dividir a árvore
}

grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV aos dados de treinamento
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest melhor acurácia com validação cruzada:", accuracy_rf)

# Random Forest melhores parâmetros
best_params = grid_search.best_params_
print("Random Forest melhores parâmetros:", best_params)

# Salva o modelo Random Forest
joblib.dump(best_rf_model, 'random_forest_model.pkl')

# Reporte de Classificação
print("Reporte de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("################################################")


################################################
############## Decision Tree ###################


from sklearn.tree import DecisionTreeClassifier

# Inicialização Decision Tree
dt_classifier = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],  # Critérios para divisão
    'max_depth': [None, 10, 20],        # Profundidade Máxima
    'min_samples_split': [2, 5, 10]     # Número mínimo para dividir a árvore
}

grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV aos dados de treinamento
grid_search.fit(X_train, y_train)
best_dt_model = grid_search.best_estimator_
y_pred_dt = best_dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree melhor acurácia com validação cruzada:", accuracy_dt)

# Decision Tree melhores parâmetros
best_params = grid_search.best_params_
print("Decision Tree melhores parâmetros:", best_params)

# Salva o modelo Decision Tree
joblib.dump(best_dt_model, 'decision_tree_model.pkl')

# Reporte de Classificação
print("Reporte de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("################################################")


################################################
############# Gradient Boosting ################

from sklearn.ensemble import GradientBoostingClassifier

# Inicialização
gb_classifier = GradientBoostingClassifier()

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],     # Número de boosting
    'learning_rate': [0.01, 0.1, 1.0],   # Taxa de Aprendizado
    'max_depth': [3, 5, 7]               # Profundidade Máxima
}

grid_search = GridSearchCV(gb_classifier, param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV aos dados de treinamento
grid_search.fit(X_train, y_train)
best_gb_model = grid_search.best_estimator_
y_pred_gb = best_gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Gradient Boosting melhor acurácia com validação cruzada:", accuracy_gb)

# Gradient Boosting melhores parâmetros
best_params = grid_search.best_params_
print("Gradient Boosting melhores parâmetros:", best_params)

# Salva o modelo Gradient Boosting
joblib.dump(best_gb_model, 'gradient_boosting_model.pkl')

# Reporte de Classificação
print("Reporte de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("################################################")

################################################
################################################
