import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
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
    input_file = 'data-mining/0-Datasets/krkoptClear_new_2.data'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names=names,        # Nome das colunas 
                     usecols=features + ['Condition'], # Define as colunas que serão utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes

    # Normalizar os dados
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    # Aplicar PCA para reduzir a dimensionalidade
    pca = PCA(n_components=2)  # Reduzido para 2 componentes principais para melhor visualização
    projected = pca.fit_transform(df_scaled)

    # Visualizar a variância explicada pelos componentes principais
    print("Variância explicada pelos componentes principais:", pca.explained_variance_ratio_)

    # Determinar o número ideal de componentes no GMM usando o Critério de Informação Bayesiano (BIC)
    n_components_range = range(1, 21)
    bic_scores = []
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n)
        gmm.fit(projected)
        bic_scores.append(gmm.bic(projected))

    optimal_n_components = n_components_range[np.argmin(bic_scores)]
    print(f"Número ideal de componentes para GMM: {optimal_n_components}")

    # Aplicar GMM com o número ótimo de componentes
    gmm = GaussianMixture(n_components=optimal_n_components)
    labels = gmm.fit_predict(projected)

    # Visualizar os resultados
    plot_samples(projected, labels, f'Clusters Labels GMM com {optimal_n_components} Componentes')

    # Codificar os rótulos reais
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(df['Condition'])

    # Calcular a acurácia
    accuracy = accuracy_score(true_labels, labels)
    print(f"Acurácia do GMM: {accuracy:.4f}")

if __name__ == "__main__":
    main()
