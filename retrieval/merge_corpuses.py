import pandas as pd

file1_path = '/home/avisingh/datasets/corpus.csv'
file2_path = '/home/avisingh/datasets/corpus_instruction_formatted_50k.csv'
output_file_path = '/home/avisingh/datasets/corpus_final.csv'

print(f"Reading first file: {file1_path}...")
df1 = pd.read_csv(file1_path)

print(f"Reading second file: {file2_path}...")
df2 = pd.read_csv(file2_path)

print("Merging the DataFrames...")
merged_df = pd.concat([df1, df2], ignore_index=True)

merged_df.to_csv(output_file_path, index=False)

total_rows = len(merged_df)

print(f"The merged data has been saved to '{output_file_path}'")
print(f"Total rows in the final file: {total_rows}")

# Size of corpus_final.csv 160646