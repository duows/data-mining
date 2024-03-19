import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    input_file = 'data-mining/0-Datasets/krkopt_missing_values_teste.data'
    output_file = 'data-mining/0-Datasets/krkoptClear_teste.data'
    
    # Carrega o conjunto de dados
    df = pd.read_csv(input_file, header=None)

    # Divide o conjunto de dados em treinamento e teste, removendo aproximadamente mil dados proporcionalmente
    train_df, test_df = train_test_split(df, test_size=26556/len(df), random_state=42)

    # Salva o conjunto de dados de treinamento sem os dados removidos
    train_df.to_csv(output_file, header=False, index=False)

if __name__ == "__main__":
    main()
