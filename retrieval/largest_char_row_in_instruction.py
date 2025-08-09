import pandas as pd

def get_max_row_char_count(file_path):
    """
    Opens a CSV file, calculates the combined character count of 'Target' and 
    'Prompt' columns for each row, and returns the highest count found.

    Args:
        file_path (str): The full path to the CSV file.

    Returns:
        None. Prints the result directly.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Ensure the required columns exist
        required_columns = ['Target', 'Prompt']
        if not all(col in df.columns for col in required_columns):
            print("Error: CSV must contain both 'Target' and 'Prompt' columns.")
            return

        # For each row, add the character lengths of 'Target' and 'Prompt'
        combined_chars_per_row = df['Target'].astype(str).str.len() + df['Prompt'].astype(str).str.len()

        # Find the maximum value from the row-wise sums
        max_char_count = combined_chars_per_row.max()

        print(f"Checked {len(df)} rows.")
        print("-" * 30)
        print(f"The highest character count for a single row is: {max_char_count}")

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Define the path to your CSV file
    csv_file_path = '/home/avisingh/datasets/training_data_v2.csv'
    
    # Run the function
    get_max_row_char_count(csv_file_path)