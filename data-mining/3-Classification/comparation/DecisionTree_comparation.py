import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, f1_score, accuracy_score

def main():
    # Carregar a base de dados de xadrez
    input_file = 'data-mining/0-Datasets/krkoptBalance.data'
    names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Distance','Condition'] 
    features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Distance']
    target = 'Condition'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas   
    
    # Selecionar os dados de entrada e saída
    x = df.loc[:, features]
    y = df.loc[:, target]

    # Dividir os dados - 70% para treinamento, 30% para teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
    
    # Criar um classificador de árvore de decisão
    clf = DecisionTreeClassifier()
    
    # Iniciar o timer
    start_time = time.time()
    
    # Treinar o classificador
    clf.fit(X_train, y_train)
    
    # Fazer previsões no conjunto de teste
    predictions = clf.predict(X_test)
    
    # Parar o timer
    

    # Plotar a árvore de decisão
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=x.columns, class_names=[str(c) for c in df['Condition'].unique()])
    plt.title("Árvore de Decisão para Condição no Xadrez")
    plt.show()

    end_time = time.time()
    
    # Calcular o tempo total
    elapsed_time = end_time - start_time
    print(f'Tempo para rodar o modelo: {elapsed_time:.2f} segundos')

    # Calcular a acurácia
    accuracy = accuracy_score(y_test, predictions) * 100
    print(f'Acurácia: {accuracy:.2f}')

    # Calcular a precisão (média ponderada)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    print(f'Precisão: {precision:.2f}')

    # Calcular o F1-Score (média ponderada)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    print(f'F1-Score: {f1:.2f}')

    total_nodes = clf.tree_.node_count
    print(f'Número total de nós: {total_nodes}')

if __name__ == "__main__":
    main()
