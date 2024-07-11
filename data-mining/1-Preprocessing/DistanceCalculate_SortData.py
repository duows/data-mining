import pandas as pd
import numpy as np

def main():
    # Faz a leitura do arquivo
    names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Condition'] 
    features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank', 'Condition']
    output_file = 'data-mining/data-mining/0-Datasets/krkoptClear_dois_dois_dois.data'
    input_file = 'data-mining/data-mining/0-Datasets/krkoptClear_dois.data'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names=names,        # Nome das colunas 
                     usecols=features,   # Define as colunas que serão  utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes


    def calculate_distance(row):
        white_king_coords = (row['White King file'], row['White King rank'])
        black_king_coords = (row['Black King file'], row['Black King rank'])
        distance = np.sqrt((black_king_coords[0] - white_king_coords[0])**2 + (black_king_coords[1] - white_king_coords[1])**2)
        return int(distance)  # Converte a distância para um inteiro
    
    # Adiciona uma nova coluna com a distância entre os reis branco e preto
    df['Distance'] = df.apply(calculate_distance, axis=1)
    
    numeric_columns = df.columns[df.columns.str.contains('file|rank|Distance')]
    df[numeric_columns] = df[numeric_columns].astype(int)

    # Reorganiza as colunas para colocar 'Distance' antes de 'Condition'
    column_order = list(df.columns)
    column_order.remove('Condition')
    column_order.insert(column_order.index('Distance') + 6, 'Condition')  # Insere 'Condition' após as colunas dos reis
    df = df[column_order]

    df.sort_values(by='Condition', inplace=True)

    # Salva o DataFrame modificado no arquivo de saída
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
