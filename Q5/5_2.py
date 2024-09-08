import faiss
import pickle
import numpy as np
from datasets import load_dataset


def mean_reciprocal_rank(ranked_results, gold_labels):
    for i, doc_id in enumerate(ranked_results, 1):
        if str(doc_id) in gold_labels: 
            return 1 / i  
    return 0 


def mean_average_precision(ranked_results, gold_labels):
    relevant_count = 0
    average_precision = 0

    for i, doc_id in enumerate(ranked_results, 1):
        if str(doc_id) in gold_labels:  
            relevant_count += 1
            average_precision += relevant_count / i

    return average_precision / len(gold_labels) if relevant_count > 0 else 0


with open("scifact_evidence_embeddings.pkl", "rb") as f:
    evidence_embeddings = pickle.load(f)

with open("scifact_claim_embeddings.pkl", "rb") as f:
    claim_embeddings = pickle.load(f)

evidence_ids = list(evidence_embeddings.keys())
evidence_vectors = np.array(list(evidence_embeddings.values())).astype('float32')

index = faiss.IndexFlatL2(evidence_vectors.shape[1])
index.add(evidence_vectors)

dataset = load_dataset("allenai/scifact", "claims")

k = 50  
total_map = 0
total_mrr = 0
num_claims = 0  


for claim_pair, claim_vector in claim_embeddings.items():
    claim_id = claim_pair[0]
    claim_vector = np.array([claim_vector]).astype('float32')
    
    gold_labels = []
    for claim in dataset['train']:
        if claim['id'] == claim_id and claim['evidence_doc_id']:
            gold_labels.append(claim['evidence_doc_id']) 
            
    if not gold_labels:
        continue  
    
    gold_labels = list(set(gold_labels))

    distances, indices = index.search(claim_vector, k)
    retrieved_doc_ids = [evidence_ids[i][0] for i in indices[0]]  

    map_score = mean_average_precision(retrieved_doc_ids, gold_labels)
    mrr_score = mean_reciprocal_rank(retrieved_doc_ids, gold_labels)
    
    total_map += map_score
    total_mrr += mrr_score

    num_claims += 1

avg_map = total_map / num_claims 
avg_mrr = total_mrr / num_claims 

print(f"Average MAP@{k}: {avg_map}")
print(f"Average MRR@{k}: {avg_mrr}")



