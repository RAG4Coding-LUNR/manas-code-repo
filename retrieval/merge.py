import csv
import os
import sys


csv.field_size_limit(sys.maxsize)

output_filename = "corpus.csv"
documentation_filename = "documentation.csv"
tutorials_filename = "tutorials.csv"

total_rows_written = 0

with open(output_filename, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    # Write the header
    writer.writerow(["ID", "Document", "Source"])

    # --- Process documentation.csv ---
    with open(documentation_filename, "r", newline="", encoding="utf-8") as doc_file:
        reader = csv.DictReader(doc_file)
        doc_id_counter = 1
        for row in reader:
            # Construct the new row
            new_row = [
                f"documentation_{doc_id_counter}",
                row["doc_content"],
                "documentation",
            ]
            writer.writerow(new_row)

            doc_id_counter += 1
            total_rows_written += 1

            # Print status every 10,000 rows
            if total_rows_written % 10000 == 0:
                print(f"Data written: {total_rows_written} rows")


    # --- Process tutorials.csv ---
    with open(tutorials_filename, "r", newline="", encoding="utf-8") as tut_file:
        reader = csv.DictReader(tut_file)
        tut_id_counter = 1
        for row in reader:
            # Construct the new row
            new_row = [
                f"online_tutorials_{tut_id_counter}",
                row["text"],
                "tutorials",
            ]
            writer.writerow(new_row)

            tut_id_counter += 1
            total_rows_written += 1

            # Print status every 10,000 rows
            if total_rows_written % 10000 == 0:
                print(f"Data written: {total_rows_written} rows")


print(f"\nProcessing complete. Total rows written: {total_rows_written}")
print(f"Data successfully written to {output_filename}")