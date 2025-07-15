import pandas as pd
import re

def filter_python_code(input_file, output_file):
    """
    Reads a CSV, extracts Python code blocks from the 'Target' column,
    and saves the result along with 'Prompt' and 'Dataset Name' to a new CSV.
    Skips rows that do not contain a Python code block.
    """
    df = pd.read_csv(input_file)
    
    # This regular expression will find all text between "```Python" and "```"
    regex = r"```Python(.*?)```"
    list_of_items = []

    counter = 0
    match_c = 0

    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        counter += 1
        if counter%10000==0:
            print(counter)
        # Use the regular expression to find all matches in the current row
        match = re.findall(regex, row['Target'], re.DOTALL)

        # If any matches were found, process the first one
        if match:
            match_c+=1
            # Extract the first match and remove leading/trailing whitespace
            extracted_code = "```Python\n" + match[0].strip() + "\n```"
            
            list_of_items.append({
                'Prompt': row['Prompt'],
                'Target': extracted_code, 
                "Dataset Name": row['Dataset Name']
            })
            
    # Create the final DataFrame from the list of dictionaries
    filtered_df = pd.DataFrame(list_of_items)
    
    # Write the filtered dataframe to a new CSV file
    filtered_df.to_csv(output_file, index=False)

    print(f"Successfully filtered the Python code with rows {match_c} and saved it to '{output_file}'")

# Specify the input and output file names
input_csv = '/home/avisingh/datasets/instruction.csv'
output_csv = '/home/avisingh/datasets/instruction_filtered_python.csv'

# Run the function
filter_python_code(input_csv, output_csv)

# Filtered Dataset Size: 1777