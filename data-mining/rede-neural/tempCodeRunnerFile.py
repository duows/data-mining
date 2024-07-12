from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import keras
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

input_file = 'data-mining/0-Datasets/krkoptBalance.data'
names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Distance','Condition'] 
features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Distance']
target = 'Condition'
df = pd.read_csv(input_file,    # Nome do arquivo com dados
                    names = names) # Nome das colunas      

x = df.loc[:, features];
y = df.loc[:, target];

# Dividir os dados - 70% para treinamento, 30% para teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

model = Sequential()
model.add(Dense(units=3, activation='relu', input_dim=7))
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
result = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Historico de treinamento')
plt.ylabel('Função de custos')
plt.xlabel('Época de treinamento')
plt.legend(['Erro treino', 'Erro teste'])
plt.show()