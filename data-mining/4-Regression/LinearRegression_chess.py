import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('Measured vs Predicted Values')
    plt.show()

def main():
    # Carrega os dados de xadrez
    input_file = 'data-mining/0-Datasets/krkoptClear_new_2.data'
    names = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance', 'Condition'] 
    features = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance']
    target = 'Condition'
    df = pd.read_csv(input_file, names=names)
    print(df.head())

    # Seleciona os dados de entrada e saída
    X = df.loc[:, features]
    y = df.loc[:, target]

    print("Total samples: {}".format(X.shape[0]))

    # Divide em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test samples: {}".format(X_test.shape[0]))

    # Normaliza os dados de entrada usando Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Cria e treina o modelo de Regressão Linear
    regr = LinearRegression()
    regr.fit(X_train, y_train)

    # Avaliação do modelo
    r2_train = regr.score(X_train, y_train)
    r2_test = regr.score(X_test, y_test)
    print('R2 no conjunto de treino: %.2f' % r2_train)
    print('R2 no conjunto de teste: %.2f' % r2_test)

    # Previsões e cálculo do erro absoluto
    y_pred = regr.predict(X_test)
    abs_error = mean_absolute_error(y_test, y_pred)
    print('Erro absoluto no conjunto de teste: %.2f' % abs_error)

    # Plot das previsões
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()
