# Importações iniciais
import time
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.svm import SVC

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Esta função imprime e plota a matriz de confusão.
    A normalização pode ser aplicada configurando `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão sem normalização')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo verdadeiro')
    plt.xlabel('Rótulo previsto')    

def main():
    # Carregar a base de dados de xadrez
    input_file = 'data-mining/0-Datasets/krkoptBalance.data'
    names = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance', 'Condition'] 
    features = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance']
    target = 'Condition'
    df = pd.read_csv(input_file, names=names)
    
    # Selecionar os dados de entrada e saída
    X = df.loc[:, features]
    y = df.loc[:, target]

    print("Total de amostras: {}".format(X.shape[0]))

    # Dividir os dados - 75% para treinamento, 25% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total de amostras de treino: {}".format(X_train.shape[0]))
    print("Total de amostras de teste: {}".format(X_test.shape[0]))

    # Normalizar os dados de entrada usando Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Iniciar o timer
    start_time = time.time()

    # Testar usando o classificador SVM do sklearn    
    svm = SVC(kernel='poly', degree=5, C=10) # poly, rbf, linear
    # Treinar usando o conjunto de dados de treino
    svm.fit(X_train, y_train)
    # Obter vetores de suporte
    print("Vetores de suporte:")
    print(svm.support_vectors_)
    # Obter índices dos vetores de suporte
    print("Índices dos vetores de suporte:")
    print(svm.support_)
    # Obter o número de vetores de suporte para cada classe
    print("Número de vetores de suporte para cada classe:")
    print(svm.n_support_)

    # Prever usando o conjunto de dados de teste
    y_hat_test = svm.predict(X_test)

    # Obter a acurácia no teste
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    precision = precision_score(y_test, y_hat_test, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro') * 100
    print("Acurácia do SVM do sk-learn: {:.2f}%".format(accuracy))
    print("Precisão do SVM do sk-learn: {:.2f}%".format(precision))
    print("F1 Score do SVM do sk-learn: {:.2f}%".format(f1))

    end_time = time.time()
    
    # Calcular o tempo total
    elapsed_time = end_time - start_time
    print(f'Tempo para rodar o modelo: {elapsed_time:.2f} segundos')

    # Obter a matriz de confusão    
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, df[target].unique(), False, "Matriz de Confusão - SVM sklearn")      
    plot_confusion_matrix(cm, df[target].unique(), True, "Matriz de Confusão - SVM sklearn normalizada")  
    plt.show()

    

if __name__ == "__main__":
    main()
