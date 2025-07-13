import sys
import csv

csv.field_size_limit(sys.maxsize)

instruction_target_file = csv.reader(open("instruction.csv","r"))
top_k_file = csv.reader(open("context_prompt.csv","r"))

print("Files loaded...")

with open("training_data.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow(["Prompt","Context","Target"])
    next(instruction_target_file)
    next(top_k_file)
    counter = 0
    for i,j in zip(instruction_target_file,top_k_file):
        
        counter+=1
        if (counter%10_000==0):
            print("Query Written: " + str(counter))

        context = ""
        
        for k in range(2,12): 
            context += str(j[k][:200]) + "\n"

        writer.writerow([str(i[0]),context,str(i[1])])

print("Files Written")