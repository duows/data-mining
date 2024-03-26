import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    
    input_file = 'data-mining-main/data-mining/0-Datasets/krkoptClear_dois.data'
    names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Condition'] 
    features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank', 'Condition']
    target = 'Condition'
    df = pd.read_csv(input_file, names=names)


    # Criação do histograma
    for feature in features:
        plt.figure(figsize=(10, 8))
        sns.histplot(df[feature], kde=True, color='blue', bins=20)
        plt.title(f'Histograma para {features}')
        plt.xlabel(feature)
        plt.ylabel('Frequência')
        plt.show()

if __name__ == "__main__":
    main()