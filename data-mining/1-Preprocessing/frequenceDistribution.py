#Data handling Imports
import pandas as pd 
import numpy as np

#Visualizing Data Imports
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():

  names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Condition'] 
  

  
  input_file = 'data-mining/0-Datasets/krkoptClear.data'

  df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas    

  frequency = df['Condition'].value_counts()

  frequency.plot(kind='bar', xlabel='Condition', ylabel='Frequency', title='Frequency Distribution of Cap_Shape')
  plt.show()

  ax = plt.axes(projection='3d')

  # # Data for a three-dimensional line
  # zline = frequency.keys()
  # xline = frequency.tolist()
  # yline = 0

  # # Data for three-dimensional scattered points
  # zdata = 15 * np.random.random(100)
  # xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
  # ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
  # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
  # plt.show()

if __name__ == "__main__":
    main()