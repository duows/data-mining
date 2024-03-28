import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    input_file = 'data-mining/0-Datasets/krkopt_missing_values_teste.data'
    output_file = 'data-mining/0-Datasets/krkoptClear_teste.data'
    
    df = pd.read_csv(input_file, header=None)

    train_df, test_df = train_test_split(df, test_size=26556/len(df), random_state=42)

    train_df.to_csv(output_file, header=False, index=False)

if __name__ == "__main__":
    main()
