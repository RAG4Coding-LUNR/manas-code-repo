import csv
import sys

csv.field_size_limit(sys.maxsize)

# The name of your CSV file
filename = "instruction.csv"
data_row_count = 0

with open(filename, 'r', encoding='utf-8') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    
    # Skip the header row
    next(csv_reader)
    
    # Loop over the remaining rows and count them
    for row in csv_reader:
        data_row_count += 1
        
print(f"The number of data rows is: {data_row_count}")