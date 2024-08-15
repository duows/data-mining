import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical, plot_model
import pydot
import graphviz

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

def plot_network_architecture(model, filename='data-mining/rede-neural/model_architecture.png'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plot_model(model, to_file='data-mining/rede-neural/model_architecture.dot', show_shapes=True, show_layer_names=True, rankdir='LR', expand_nested=True)
    try:
        (graph,) = pydot.graph_from_dot_file('data-mining/rede-neural/model_architecture.dot')
        graph.set_dpi(100)
        graph.set_graph_defaults(dpi=100)
        graph.write_png(filename)
        print(f"Arquitetura do modelo salva em '{filename}'.")
    except Exception as e:
        print(f"Erro ao gerar o diagrama da arquitetura: {e}")

def main():
    input_file = 'data-mining/0-Datasets/krkoptClear_new_2.data'
    names = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance', 'Condition'] 
    features = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance']
    target = 'Condition'
    df = pd.read_csv(input_file, names=names)

    X = df.loc[:, features]
    y = df.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(units=128, input_dim=X_train.shape[1], kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.2))

    for _ in range(5):
        model.add(Dense(units=50, kernel_regularizer=l2(0.001)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.2))

    model.add(Dense(units=y_train_cat.shape[1], activation='softmax'))

    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(X_train, y_train_cat, epochs=500, batch_size=32, validation_data=(X_test, y_test_cat))
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'Tempo para rodar o modelo: {elapsed_time:.2f} segundos')

    plot_training_history(history)
    plot_network_architecture(model)

    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print('Erro no conjunto de teste: {:.2f}'.format(loss))
    print('Acurácia no conjunto de teste: {:.2f}%'.format(accuracy * 100))

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_cat, axis=1)

    print(classification_report(y_test_classes, y_pred_classes))

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_test_classes, y_pred_classes, average='macro')

    print(f'Acurácia: {accuracy * 100:.2f}%')
    print(f'Precisão: {precision * 100:.2f}%')
    print(f'F1-Score: {f1 * 100:.2f}%')

if __name__ == "__main__":
    main()
