import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Erro treino')
    plt.plot(history.history['val_loss'], label='Erro teste')
    plt.plot(history.history['accuracy'], label='Acurácia treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia teste')
    plt.title('Histórico de Treinamento')
    plt.ylabel('Função de custo / Acurácia')
    plt.xlabel('Época de treinamento')
    plt.legend()
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test samples: {}".format(X_test.shape[0]))

    # Normaliza os dados de entrada usando Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Transforma os labels em one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Cria e treina o modelo de Rede Neural
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=y_train.shape[1], activation='softmax'))  # Para classificação multiclasse

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

    # Plot do histórico de treinamento
    plot_training_history(history)

    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Erro no conjunto de teste: {:.2f}'.format(loss))
    print('Acurácia no conjunto de teste: {:.2f}%'.format(accuracy * 100))

if __name__ == "__main__":
    main()
