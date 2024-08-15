import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Definindo a função KMeans do zero
def KMeans_scratch(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    # Escolhendo centróides aleatoriamente
    centroids = x[idx, :]  # Etapa 1

    # Iterando para ajustar os centróides
    for _ in range(no_of_iterations):
        # Calculando as distâncias de cada ponto para os centróides
        distances = cdist(x, centroids, 'euclidean')  # Etapa 2

        # Atribuindo cada ponto ao centróide mais próximo
        points = np.array([np.argmin(i) for i in distances])  # Etapa 3

        # Recalculando os centróides com base na média dos pontos atribuídos
        centroids = np.array([x[points == idx].mean(axis=0) for idx in range(k)])  # Etapa 4

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
    input_file = 'data-mining/0-Datasets/krkoptClear_new_2.data'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names=names,        # Nome das colunas 
                     usecols=features + ['Condition'], # Define as colunas que serão utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes

    # Transformar os dados usando PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(df[features])
    print("Variância explicada pelos componentes principais:", pca.explained_variance_ratio_)

    # Plotar os dados originais
    plot_samples(projected, df['Condition'], 'Etiquetas Originais')

    # Aplicando nossa função KMeans do zero
    labels_scratch = KMeans_scratch(projected, 6, 100)
    
    # Visualizar os resultados do KMeans do zero
    plot_samples(projected, labels_scratch, 'Etiquetas dos Clusters KMeans do zero')

    # Aplicando KMeans do sklearn
    kmeans = KMeans(n_clusters=6, n_init=10, max_iter=300, random_state=42).fit(projected)
    print("Inércia do KMeans:", kmeans.inertia_)
    
    # Calculando a pontuação do silhouette
    score = silhouette_score(projected, kmeans.labels_)
    print("Para n_clusters = {}, a pontuação do silhouette é {})".format(6, score))

    # Visualizar os resultados do KMeans do sklearn
    plot_samples(projected, kmeans.labels_, 'Etiquetas dos Clusters KMeans do sklearn')

    plt.show()

if __name__ == "__main__":
    main()
