import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def group_condition_intervals(condition):
    if 0 <= condition <= 3:
        return '0-3'
    elif 4 <= condition <= 7:
        return '4-7'
    elif 8 <= condition < 10:
        return '8-10'
    elif 11 <= condition < 14:
        return '11-14'
    elif 15 <= condition < 17:
        return '15-17'
    elif 18:
        return '18'
    else:
        return 'Other'

def main():
    # Faz a leitura do arquivo
    input_file = 'data-mining/0-Datasets/krkoptClear_new_2.data'
    output_file = 'data-mining/0-Datasets/krkoptBalance.data'
    names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Condition'] 
    features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank']
    target = 'Condition'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")

    # Agrupa a coluna Condition em intervalos
    df['Condition Interval'] = df['Condition'].apply(group_condition_intervals)
    ShowInformationDataFrame(df,"Dataframe com Intervalos de Condition")

    # Separating out the features
    x = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:,[target]].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data = x, columns = features)
    normalizedDf = pd.concat([normalizedDf.reset_index(drop=True), df[[target, 'Condition Interval']].reset_index(drop=True)], axis = 1)
    ShowInformationDataFrame(normalizedDf,"Dataframe Normalized")

    # PCA projection
    pca = PCA()    
    principalComponents = pca.fit_transform(x)
    print("Explained variance per component:")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n")

    principalDf = pd.DataFrame(data = principalComponents[:,0:2], 
                               columns = ['principal component 1', 
                                          'principal component 2'])
    finalDf = pd.concat([principalDf.reset_index(drop=True), df[[target, 'Condition Interval']].reset_index(drop=True)], axis = 1)    
    ShowInformationDataFrame(finalDf,"Dataframe PCA")
    
    VisualizePcaProjection(finalDf, 'Condition Interval')

    df.to_csv(output_file, header=False, index=False)

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")
             
def VisualizePcaProjection(finalDf, targetColumn):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = finalDf[targetColumn].unique()
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

if __name__ == "__main__":
    main()
