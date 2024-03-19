import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    # Faz a leitura do arquivo
    input_file = 'data-mining/0-Datasets/krkoptClear_dois.data'
    names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Condition'] 
    features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank']
    target = 'Condition'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")

    # Separating out the features
    x = df.loc[:, features].values
    
    # Separating out the target
    y = df.loc[:,[target]].values

    # Z-score normalization
    #x_zcore = StandardScaler().fit_transform(x)
    #normalized1Df = pd.DataFrame(data = x_zcore, columns = features)
    #normalized1Df = pd.concat([normalized1Df, df[[target]]], axis = 1)
    #ShowInformationDataFrame(normalized1Df,"Dataframe Z-Score Normalized")

    x_zcore = StandardScaler().fit_transform(x)
    normalized1Df = pd.DataFrame(data = x_zcore, columns = features)
    normalized1Df = pd.concat([normalized1Df, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalized1Df,"Dataframe Z-Score Normalized")

    # Mix-Max normalization
    # Mix-Max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data=x_minmax, columns=features)
    normalized2Df[target] = df[target].values  # Adicione esta linha
    ShowInformationDataFrame(normalized2Df, "Dataframe Min-Max Normalized")

    # Plotando a matriz de correlação min-max
    correlation_matrix = normalized2Df.corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Matriz de Correlação Min-Max")
    plt.show()

    def VisualizePcaProjection(finalDf, targetColumn):
        fig = plt.figure(figsize = (20,12))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        #0, 1, 2, 
        colors = ['r', 'g', 'b', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown', 'pink', 'lime', 'gray', 'olive', 'teal', 'navy', 'maroon', 'fuchsia']

        for target, color in zip(targets,colors):
            indicesToKeep = finalDf[targetColumn] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                    finalDf.loc[indicesToKeep, 'principal component 2'],
                    c = color, s = 50)
        ax.legend(targets)
        ax.grid()
        plt.show()

    def VisualizePcaProjection3D(finalDf, targetColumn):
        fig = plt.figure(figsize=(30, 20))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_zlabel('Principal Component 3', fontsize=15)
        ax.set_title('3 component PCA', fontsize=20)
        targets = finalDf[targetColumn].unique()
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown', 'pink', 'lime', 'gray', 'olive', 'teal',
                'navy', 'maroon', 'fuchsia']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf[targetColumn] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                    finalDf.loc[indicesToKeep, 'principal component 2'],
                    finalDf.loc[indicesToKeep, 'principal component 3'],
                    c=color, s=50, label=target)
        ax.legend()
        ax.grid()
        plt.show()
    
    pca = PCA(n_components=3)  # Set the number of components to 3
    principalComponents = pca.fit_transform(x_zcore)

    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2', 'principal component 3'])

    finalDf = pd.concat([principalDf, df[[target]]], axis=1)

    VisualizePcaProjection3D(finalDf, target)

    # pca = PCA()
    # principalComponents = pca.fit_transform(x_zcore)
    # print('Explained variance ratio:')
    # print(pca.explained_variance_ratio_.tolist())
    # print(x_zcore)
    
    # principalDf = pd.DataFrame(data = principalComponents[:, 0:2], columns = ['principal component 1', 'principal component 2'])

    # finalDf = pd.concat([principalDf, df[[target]]], axis = 1)
    # finalDf.describe()

    # VisualizePcaProjection(finalDf, target)

    print(df[target].value_counts())

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n") 

#x_minmax = MinMaxScaler().fit_transform(x) 
#normalized_minmax = pd.DataFrame(x_minmax, columns = features) 
#normalized_minmax = pd.concat([normalized_minmax, df[[target]]], axis = 1) 
#normalized_minmax.describe()

#plot correlation matrix


if __name__ == "__main__":
    main()