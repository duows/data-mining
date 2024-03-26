import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd 

    # Faz a leitura do arquivo
input_file = 'data-mining-main/data-mining/0-Datasets/krkoptClear.data'
names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Condition'] 
features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank']
target = 'Condition'
df = pd.read_csv(input_file,    # Nome do arquivo com dados
                    names = names) # Nome das colunas    


# Separating out the features
x = df.loc[:, features].values

    # Separating out the target
y = df.loc[:,[target]].values


plt.title("Line graph")  
plt.xlabel("X axis")  
plt.ylabel("Y axis")  
plt.plot(x, y, color ="red")  
plt.show()