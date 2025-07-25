import argparse
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import csv
import sys
import time
import torch
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# By default using "codesage/codesage-base-v2", /home/avisingh/retreival
# Command: retrieval.py --input_query_csv instruction.csv --input_corpus_csv retreival_docs_wo_solutions.csv

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="codesage/codesage-base-v2", help="sentence-transformer model for the retriever LM.")
parser.add_argument('--input_corpus_csv', default="/home/avisingh/datasets/corpus_final_chunked.csv", help="Input CSV file to use as corpus for retriever.")
parser.add_argument('--input_query_csv', default="/home/avisingh/datasets/training_data_sampled_10k.csv", help="Input CSV file to use as queries for retriever.")
parser.add_argument('--top_k', default=10, help="Number of documents to retrieve.")
parser.add_argument('--output_csv', default="/home/avisingh/datasets/context_prompt_v2.csv", help="CSV file to save the output in.")
parser = parser.parse_args()

start_time = time.perf_counter()

embedder = SentenceTransformer(parser.model, trust_remote_code=True, device='cuda', model_kwargs={"torch_dtype": torch.float16})

print("Loaded embedding model...")

corpus = []
with open(parser.input_corpus_csv, "r", newline='') as f:
    reader = csv.DictReader(f)
    for row in tqdm(iterable=reader,total=477591):
        corpus.append(row["Document"])

# 10k from the  50k from instruction.csv

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True,batch_size=512)

print("Embedded corpus...")

queries = []
with open(parser.input_query_csv, "r", newline='') as f:
    reader = csv.DictReader(f)
    for row in tqdm(iterable=reader,total=10000):
        queries.append(row["Prompt"])

all_res = []
query_embeddings = embedder.encode(queries, convert_to_tensor=True, show_progress_bar=True,batch_size=512)
print("Embedded queries...")

print("Performing semantic search...")
for i, (query,query_embedding) in enumerate(zip(queries, query_embeddings)):
    if (i%1000==0):
        print("Finished Query: " + str(i) + "/10000")
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=int(parser.top_k))
    hits = hits[0]      #Get the hits for the first query
    res = {"id":i, "query":query}
    for j, hit in enumerate(hits):
        col_name = "rank_" + str(j)
        res[col_name] = corpus[hit['corpus_id']]
    all_res.append(res)


print("Performed semantic search...")

fieldname = ["id","query"]
for i, hit in enumerate(hits):
    col_name = "rank_" + str(i)
    fieldname.append(col_name)

with open(parser.output_csv,"w") as file:
    writer = csv.DictWriter(file,fieldnames=fieldname)
    writer.writeheader()
    writer.writerows(all_res)

print(f"CSV File with Retrievals written to {parser.output_csv}")

end_time = time.perf_counter()

# 4. Calculate the elapsed time
duration = end_time - start_time

print(f"The code took {duration:.4f} seconds to run.")

# Took 9407.3922 seconds to run