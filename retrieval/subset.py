import pandas as pd

# Load the dataset
df = pd.read_csv('/home/avisingh/datasets/instruction_filtered_all.csv')

# --- First Sample (10k rows with all columns) ---

# Randomly sample 10,000 rows for the first file.
training_sample_df = df.sample(n=10000, random_state=42)

# Save the first sampled data (all columns) to a new CSV file.
training_sample_df.to_csv('/home/avisingh/datasets/training_data_sampled_10k.csv', index=False)

print("Successfully sampled 10,000 rows and saved to 'training_data_sampled_10k.csv'")

# --- Second, Formatted Sample (50k rows) ---

# Create a new DataFrame that excludes the rows already sampled.
remaining_df = df.drop(training_sample_df.index)

# From the remaining data, sample 50,000 exclusive rows.
exclusive_sample_df = remaining_df.sample(n=50000, random_state=42)

# Start with just the 'Target' column from the 50k sample.
# Use .copy() to ensure modifications don't affect other DataFrames.
final_df = exclusive_sample_df[['Target']].copy()

# Rename the 'Target' column to 'Document'.
final_df.rename(columns={'Target': 'Document'}, inplace=True)

# Add the 'Source' column with the static value 'instructions'.
final_df['Source'] = 'instructions'

# The sample has a random index, so we reset it to a clean 0-49999 sequence.
final_df.reset_index(drop=True, inplace=True)

# Add the 'ID' column using the new, clean index.
final_df['ID'] = 'instruction_' + (final_df.index).astype(str)

# Reorder the columns into the final desired format: ID, Document, Source.
final_df = final_df[['ID', 'Document', 'Source']]

# Save the fully formatted DataFrame to a new CSV file.
final_df.to_csv('/home/avisingh/datasets/corpus_instruction_formatted_50k.csv', index=False)

print("Successfully created formatted 50k sample and saved to 'validation_data_formatted_50k.csv'")