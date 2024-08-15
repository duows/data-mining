import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
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

    # Iniciar o timer
    start_time = time.time()

    # Normaliza os dados de entrada usando Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Transforma os labels em one-hot encoding
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))  # Taxa de dropout ajustada

    model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))  # Taxa de dropout ajustada

    model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))  # Taxa de dropout ajustada
    
    model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))  # Taxa de dropout ajustada

    model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))  # Taxa de dropout ajustada

    model.add(Dense(units=y_train_cat.shape[1], activation='softmax'))  # Para classificação multiclasse

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Adiciona Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train_cat, epochs=200, batch_size=32, validation_data=(X_test, y_test_cat), callbacks=[early_stopping])

    end_time = time.time()
    
    # Calcular o tempo total
    elapsed_time = end_time - start_time
    print(f'Tempo para rodar o modelo: {elapsed_time:.2f} segundos')

    # Plot do histórico de treinamento
    plot_training_history(history)

    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print('Erro no conjunto de teste: {:.2f}'.format(loss))
    print('Acurácia no conjunto de teste: {:.2f}%'.format(accuracy * 100))

    # Previsões no conjunto de teste
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_cat, axis=1)

    # Relatório de classificação
    print(classification_report(y_test_classes, y_pred_classes))

if __name__ == "__main__":
    main()
