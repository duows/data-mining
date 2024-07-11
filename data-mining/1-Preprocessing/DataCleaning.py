import pandas as pd
import numpy as np

# Função para substituir letras por números
def replace_letters_with_numbers(df):
    for char in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        df.replace(regex=rf'^{char}$', value=str(ord(char) - ord('a') + 1), inplace=True)
    return df

def convert_condition_to_integers(df):
    condition_mapping = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, 
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "draw": 18
    }
    df['Condition'] = df['Condition'].map(condition_mapping)

def main():
    # Faz a leitura do arquivo
    names = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank','Condition'] 
    features = ['White King file','White King rank','White Rook file','White Rook rank','Black King file','Black King rank', 'Condition']
    output_file = 'data-mining/0-Datasets/krkoptClear_new.data'
    input_file = 'data-mining/0-Datasets/krkopt_missing_values.data'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes
    
    df_original = df.copy()
    # Imprime as 15 primeiras linhas do arquivo
    print("PRIMEIRAS 15 LINHAS\n")
    print(df.head(15))
    print("\n")        

    # Imprime informações sobre dos dados
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")
    
    # Imprime uma analise descritiva sobre dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")
    
    # Imprime a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES\n")
    print(df.isnull().sum())
    print("\n")    
    
    columns_missing_value = df.columns[df.isnull().any()]
    print(columns_missing_value)
    method = 'delete' # number or median or mean or mode
    
    for c in columns_missing_value:
        UpdateMissingValues(df, c, method)
    
    print(df.describe())
    print("\n")
    print(df.head(15))
    print(df_original.head(15))
    print("\n")
    
    # Substitui letras por números
    df = replace_letters_with_numbers(df)
    
    # Converte nomes escritos por extenso em números inteiros na coluna 'Condition'
    convert_condition_to_integers(df)
    
    # Substitui 'draw' por -1 na coluna 'Condition'
    # df['Condition'].replace('draw', -1, inplace=True)
    
    # Salva arquivo com o tratamento para dados faltantes
    df.to_csv(output_file, header=False, index=False)  

def UpdateMissingValues(df, column, method="mode", number=0):
    if method == 'number':
        # Substituindo valores ausentes por um número
        df[column].fillna(number, inplace=True)
    elif method == 'median':
        # Substituindo valores ausentes pela mediana 
        median = df[column].median()
        df[column].fillna(median, inplace=True)
    elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)
    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df[column].mode()[0]
        df[column].fillna(mode, inplace=True)
    elif method == 'delete':
        #delete a full row with missing values
        df.dropna(axis=0, how='any', inplace=True)

if __name__ == "__main__":
    main()
