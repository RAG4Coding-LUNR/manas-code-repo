import pandas as pd
import re

def filter_python_code(input_file, output_file):
    df = pd.read_csv(input_file)
    # Create a new, empty dataframe to store our results
    filtered_df = pd.DataFrame(columns=['Prompt','Target','Dataset Name'])

    # This regular expression will find all text between "```Python" and "```"
    regex = r"```Python(.*?)```"
    list_of_items = []

    # Loop through each row in the 'Target' column of the dataframe
    for index, row in df.iterrows():
        # Use the regular expression to find all matches in the current row
        match = re.findall(regex, row['Target'], re.DOTALL) # Theres only one match

        # If any matches were found, add them to our new dataframe
        if match:
            match = match[0].strip()
            list_of_items.append({
                'Prompt':row['Prompt'],
                'Target':match,
                "Dataset Name":row['Dataset Name']
                })
            
    filtered_df = pd.DataFrame(list_of_items, columns=['Prompt','Target','Dataset Name'])
    # Write the filtered dataframe to a new CSV file
    filtered_df.to_csv(output_file, index=False)

    print(f"Successfully filtered the Python code and saved it to '{output_file}'")

# Specify the input and output file names
input_csv = 'instruction.csv'
output_csv = 'instruction_filtered_python.csv'

# Run the function
filter_python_code(input_csv, output_csv)