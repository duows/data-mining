# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

def main():
    # Carregar a base de dados de xadrez
    input_file = 'data-mining/0-Datasets/krkoptClear_new_2.data'
    names = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance', 'Condition'] 
    features = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance']
    target = 'Condition'
    df = pd.read_csv(input_file, names=names)
    
    # Selecionar os dados de entrada e saída
    X = df.loc[:, features]
    y = df.loc[:, target]

    print("Total samples: {}".format(X.shape[0]))

    # Dividir os dados - 75% para treinamento, 25% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Normalizar os dados de entrada usando Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Testar usando o classificador SVM do sklearn    
    svm = SVC(kernel='poly') # poly, rbf, linear
    # Treinar usando o conjunto de dados de treino
    svm.fit(X_train, y_train)
    # Obter vetores de suporte
    print("Support Vectors:")
    print(svm.support_vectors_)
    # Obter índices dos vetores de suporte
    print("Indices of Support Vectors:")
    print(svm.support_)
    # Obter o número de vetores de suporte para cada classe
    print("Number of Support Vectors for Each Class:")
    print(svm.n_support_)

    # Prever usando o conjunto de dados de teste
    y_hat_test = svm.predict(X_test)

    # Obter a acurácia no teste
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy SVM from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score SVM from sk-learn: {:.2f}%".format(f1))

    # Obter a matriz de confusão    
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, df[target].unique(), False, "Confusion Matrix - SVM sklearn")      
    plot_confusion_matrix(cm, df[target].unique(), True, "Confusion Matrix - SVM sklearn normalized")  
    plt.show()

if __name__ == "__main__":
    main()
