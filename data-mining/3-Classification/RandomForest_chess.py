from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def main():
    # Carregar a base de dados de xadrez
    input_file = 'data-mining/0-Datasets/krkoptClear_new_2.data'
    names = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance', 'Condition'] 
    features = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance']
    target = 'Condition'
    df = pd.read_csv(input_file, names = names)
    
    # Selecionar os dados de entrada e saída
    x = df.loc[:, features]
    y = df.loc[:, target]

    # Dividir os dados - 70% para treinamento, 30% para teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
    
    # Criar um classificador RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    
    # Treinar o classificador
    clf.fit(X_train, y_train)
    
    # Fazer previsões
    predictions = clf.predict(X_test)
    print('Predictions:')
    print(predictions)

    # Avaliar a acurácia
    result = clf.score(X_test, y_test)
    print('Accuracy:')
    print(result)
    
    # Plotar a importância das características
    feature_importances = clf.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance in RandomForest Classifier')
    plt.show()

if __name__ == "__main__":
    main()
