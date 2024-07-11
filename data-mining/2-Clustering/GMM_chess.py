import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def plot_samples(projected, labels, title):
    plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i, 0], projected[labels == i, 1], label=i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    plt.title(title)
    plt.show()

def main():
    # Carregar a base de dados de xadrez
    names = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance', 'Condition']
    features = ['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank', 'Distance']
    input_file = 'data-mining/0-Datasets/krkoptClear_dois_dois_dois.data'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features + ['Condition'], # Define as colunas que serão utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes

    # Transformar os dados usando PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(df[features])

    # Visualizar a variância explicada pelos componentes principais
    print("Variância explicada pelos componentes principais:", pca.explained_variance_ratio_)

    # Aplicando Gaussian Mixture Model
    gmm = GaussianMixture(n_components=6)
    labels = gmm.fit_predict(projected)

    # Visualizar os resultados
    plot_samples(projected, labels, 'Clusters Labels GMM')

if __name__ == "__main__":
    main()
