import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Calculate distance between two points
def minkowski_distance(a, b, p=1):    
    dim = len(a)    
    distance = 0
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
    distance = distance**(1/p)    
    return distance

def knn_predict(X_train, X_test, y_train, k, p):    
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]
        counter = Counter(y_train[df_nn.index])
        prediction = counter.most_common()[0][0]
        y_hat_test.append(prediction)
        
    return y_hat_test

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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

    # Normalizar os dados de entrada usando Z-score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Definir a validação cruzada
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    
    # STEP 1 - TESTS USING knn classifier written from scratch
    y_hat_all = []
    y_true_all = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_hat = knn_predict(X_train, X_test, pd.Series(y_train), k=5, p=2)
        y_hat_all.extend(y_hat)
        y_true_all.extend(y_test)
    
    accuracy = accuracy_score(y_true_all, y_hat_all) * 100
    f1 = f1_score(y_true_all, y_hat_all, average='macro')
    print("Accuracy K-NN from scratch: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from scratch: {:.2f}%".format(f1))
    
    cm = confusion_matrix(y_true_all, y_hat_all)
    plot_confusion_matrix(cm, df[target].unique(), False, "Confusion Matrix - K-NN from scratch")
    plot_confusion_matrix(cm, df[target].unique(), True, "Confusion Matrix - K-NN from scratch normalized")

    # STEP 2 - TESTS USING knn classifier from sk-learn
    y_hat_all = []
    y_true_all = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_hat = knn.predict(X_test)
        y_hat_all.extend(y_hat)
        y_true_all.extend(y_test)
    
    accuracy = accuracy_score(y_true_all, y_hat_all) * 100
    f1 = f1_score(y_true_all, y_hat_all, average='macro')
    print("Accuracy K-NN from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from sk-learn: {:.2f}%".format(f1))

    cm = confusion_matrix(y_true_all, y_hat_all)
    plot_confusion_matrix(cm, df[target].unique(), False, "Confusion Matrix - K-NN sklearn")
    plot_confusion_matrix(cm, df[target].unique(), True, "Confusion Matrix - K-NN sklearn normalized")
    plt.show()

if __name__ == "__main__":
    main()
