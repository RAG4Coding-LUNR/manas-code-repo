import pandas as pd

def analyze_document_length(file_path):
    df = pd.read_csv(file_path)
    df['char_count'] = df['Document'].str.len()
    return df['char_count'].describe()[['25%', '50%', '75%','max']]


if __name__ == "__main__":
    print(analyze_document_length("/home/avisingh/datasets/corpus_final_chunked.csv"))