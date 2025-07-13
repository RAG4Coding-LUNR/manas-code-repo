import pandas as pd

# Load the dataset
df = pd.read_csv('training_data.csv')

# Randomly sample 10,000 rows
sampled_df = df.sample(n=10000)

# Save the sampled data to a new CSV file
sampled_df.to_csv('training_data_sampled.csv', index=False)

print("Successfully sampled 10,000 rows and saved to 'training_data_sampled.csv'")