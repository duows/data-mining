# Importações iniciais
import time
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Calcular a distância entre dois pontos
def minkowski_distance(a, b, p=1):    
    # Armazenar o número de dimensões
    dim = len(a)    
    # Definir a distância inicial como 0
    distance = 0
    
    # Calcular a distância de Minkowski usando o parâmetro p
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
        
    distance = distance**(1/p)    
    return distance

def knn_predict(X_train, X_test, y_train, k, p):    
    # Fazer previsões nos dados de teste
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        # Armazenar distâncias em um dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        
        # Ordenar distâncias e considerar apenas os k pontos mais próximos
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Criar objeto Counter para rastrear os rótulos dos k vizinhos mais próximos
        counter = Counter(y_train[df_nn.index])

        # Obter o rótulo mais comum de todos os vizinhos mais próximos
        prediction = counter.most_common()[0][0]
        
        # Adicionar previsão à lista de saída
        y_hat_test.append(prediction)
        
    return y_hat_test

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

    # STEP 1 - TESTES USANDO O CLASSIFICADOR K-NN IMPLEMENTADO DO ZERO    
    # Fazer previsões no conjunto de dados de teste usando o classificador K-NN
    y_hat_test = knn_predict(X_train, X_test, y_train, k=5, p=2)

    # Obter a acurácia no teste
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    precision = precision_score(y_test, y_hat_test, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Acurácia K-NN implementado do zero: {:.2f}%".format(accuracy))
    print("Precisão K-NN implementado do zero: {:.2f}%".format(precision))
    print("F1 Score K-NN implementado do zero: {:.2f}%".format(f1))

    # Obter a matriz de confusão
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, df[target].unique(), False, "Matriz de Confusão - K-NN implementado do zero")      
    plot_confusion_matrix(cm, df[target].unique(), True, "Matriz de Confusão - K-NN implementado do zero normalizada")  

    # STEP 2 - TESTES USANDO O CLASSIFICADOR K-NN DO SK-LEARN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_hat_test = knn.predict(X_test)

    # Obter a acurácia no teste
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    precision = precision_score(y_test, y_hat_test, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Acurácia K-NN do sk-learn: {:.2f}%".format(accuracy))
    print("Precisão K-NN do sk-learn: {:.2f}%".format(precision))
    print("F1 Score K-NN do sk-learn: {:.2f}%".format(f1))

    end_time = time.time()
    
    # Calcular o tempo total
    elapsed_time = end_time - start_time
    print(f'Tempo para rodar o modelo: {elapsed_time:.2f} segundos')

    # Obter a matriz de confusão
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, df[target].unique(), False, "Matriz de Confusão - K-NN sklearn")      
    plot_confusion_matrix(cm, df[target].unique(), True, "Matriz de Confusão - K-NN sklearn normalizada")  
    plt.show()

if __name__ == "__main__":
    main()
