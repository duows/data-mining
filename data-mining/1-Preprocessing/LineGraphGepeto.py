import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    
    input_file = 'data-mining-main/data-mining/0-Datasets/krkoptClear.data'
    names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Condition'] 
    features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank', 'Condition']
    target = 'Condition'
    df = pd.read_csv(input_file, names=names)


    x = df.loc[:, features].values
    plt.figure(figsize=(8, 6))
    for i in range(x.shape[0]):
        plt.plot(range(x.shape[1]), x[i], label=f'Amostra {i+1}')
    plt.title('Gráfico de Linha da Matriz Numpy')
    plt.xlabel('Índice da Feature')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()