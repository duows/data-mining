from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

def main():
    # Carregar a base de dados de xadrez
    input_file = 'data-mining/0-Datasets/krkoptBalance.data'
    names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Distance','Condition'] 
    features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Distance']
    target = 'Condition'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas   
    
    # Selecionar os dados de entrada e saída
    # X = df[['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance']]
    # y = df['Condition']
    
    x = df.loc[:, features];
    y = df.loc[:, target];

    # Dividir os dados - 70% para treinamento, 30% para teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
    
    # Criar um classificador de árvore de decisão
    clf = DecisionTreeClassifier(max_leaf_nodes=3)
    
    # Treinar o classificador
    clf.fit(X_train, y_train)
    
    # Plotar a árvore de decisão
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=x.columns, class_names=[str(c) for c in df['Condition'].unique()])
    plt.title("Árvore de Decisão para Condição no Xadrez")
    plt.show()

    predictions = clf.predict(X_test)
    print(predictions)

    result = clf.score(X_test, y_test)
    print('Acuraccy:')
    print(result)
if __name__ == "__main__":
    main()
