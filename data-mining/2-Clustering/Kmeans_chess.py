import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Definindo nossa função kmeans do zero
def KMeans_scratch(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    # Escolhendo centróides aleatoriamente
    centroids = x[idx, :]  # Etapa 1

    # Encontrando a distância entre os centróides e todos os pontos de dados
    distances = cdist(x, centroids, 'euclidean')  # Etapa 2

    # Centróide com a distância mínima
    points = np.array([np.argmin(i) for i in distances])  # Etapa 3

    # Repetindo as etapas acima por um número definido de iterações
    # Etapa 4
    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Atualizando centróides tomando a média do cluster a que pertencem
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)  # Centrós atualizados

        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points

def plot_samples(projected, labels, title):
    fig = plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i, 0], projected[labels == i, 1], label=i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    plt.title(title)

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
    pca = PCA(2)
    projected = pca.fit_transform(df[features])
    print(pca.explained_variance_ratio_)
    print(df[features].shape)
    print(projected.shape)
    plot_samples(projected, df['Condition'], 'Etiquetas Originais')

    # Aplicando nossa função kmeans do zero
    labels = KMeans_scratch(projected, 6, 5)
    
    # Visualizar os resultados
    plot_samples(projected, labels, 'Etiquetas dos Clusters KMeans do zero')

    # Aplicando função kmeans do sklearn
    kmeans = KMeans(n_clusters=6).fit(projected)
    print(kmeans.inertia_)
    centers = kmeans.cluster_centers_
    score = silhouette_score(projected, kmeans.labels_)
    print("Para n_clusters = {}, a pontuação do silhouette é {})".format(6, score))

    # Visualizar os resultados sklearn
    plot_samples(projected, kmeans.labels_, 'Etiquetas dos Clusters KMeans do sklearn')

    plt.show()

if __name__ == "__main__":
    main()
