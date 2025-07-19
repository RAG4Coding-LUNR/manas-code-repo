import pandas as pd
import re

def chunk_document(file_path, median_char):
    df = pd.read_csv(file_path)
    ls = []
    for _,i in df.iterrows():
        chunks = re.findall('.{1,'+ str(median_char) +'}', i.loc['Document'],flags=re.DOTALL)
        len_of_val = len(chunks)
        
        if len_of_val == 1: 
            ls.append(i.tolist())
            continue

        for j in range(len_of_val):
            ls.append([i.loc['ID']+"_"+str(j),chunks[j],i.loc['Source']])
        
    df_other = pd.DataFrame(ls,columns=['ID', 'Document', 'Source'])
    df_other.to_csv("/home/avisingh/datasets/corpus_final_chunked.csv",index=False)

if __name__ == "__main__":
    chunk_document("/home/avisingh/datasets/corpus_final.csv",1370)