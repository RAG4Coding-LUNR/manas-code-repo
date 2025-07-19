import sys
import csv

csv.field_size_limit(sys.maxsize)

instruction_target_file = csv.reader(open("/home/avisingh/datasets/training_data_sampled_10k.csv","r"))
top_k_file = csv.reader(open("/home/avisingh/datasets/context_prompt_v2.csv","r"))

'''
This merges the Query, Target from instructions with context 
'''

print("Files loaded...")

with open("/home/avisingh/datasets/training_data_v2.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow(["Prompt","Context","Target"])
    next(instruction_target_file)
    next(top_k_file)
    counter = 0
    for i,j in zip(instruction_target_file,top_k_file):
        
        counter+=1
        if (counter%1000==0):
            print("Query Written: " + str(counter))

        context = "<context>"
        
        for k in range(2,12): 
            context += "Doc" + str(k-1) + ": " + str(j[k]) + "\n"

        context = context + "</context>"

        writer.writerow([str(i[0]),context,str(i[1])])

print("Files Written")