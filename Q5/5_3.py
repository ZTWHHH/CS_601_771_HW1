import pickle
import numpy as np
from datasets import load_dataset
from elasticsearch import Elasticsearch


CLOUD_ID = "6d448564dea242e6ba3422425ef97ffd:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGJmNTlkOTY5MmJlYjQyMDRiZDQ2ZWU3Y2M0NmY2OWRlJDJmMmJlZTFiYzMxYjQyN2Y5MWQ1Mzg2M2U4NmExZmUy"
API_KEY = "NDd3YXlaRUJHUUhlczJ3bVpyQkg6Vkh5bWFLTFlUZjZmVnB3WDI4RXBQdw=="

es = Elasticsearch(
    cloud_id=CLOUD_ID,
    api_key=API_KEY
)

if not es.ping():
    raise ValueError("Connection failed")

with open("scifact_evidence_embeddings.pkl", "rb") as f:
    evidence_embeddings = pickle.load(f)

with open("scifact_claim_embeddings.pkl", "rb") as f:
    claim_embeddings = pickle.load(f)

dataset = load_dataset("allenai/scifact", "claims")

for evidence_pair, evidence_vector in evidence_embeddings.items():
    evidence_id, evidence_text = evidence_pair
    es.index(index='documents', id=evidence_id, body={'text': evidence_text})


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


k = 10
total_map = 0
total_mrr = 0
num_claims = 0  


for claim_pair, claim_vector in claim_embeddings.items():
    claim_id, claim_text = claim_pair

    gold_labels = []
    for claim in dataset['train']:
        if claim['id'] == claim_id and claim['evidence_doc_id']:
            gold_labels.append(claim['evidence_doc_id']) 

    if not gold_labels:
        continue

    gold_labels = list(set(gold_labels))

    response = es.search(
    index='documents',
    body={
        "query": {
            "match": {
                "text": claim_text  
            }
        }
    },
    size=k)

    retrieved_doc_ids = [hit['_id'] for hit in response['hits']['hits']]

    map_score = mean_average_precision(retrieved_doc_ids, gold_labels)
    mrr_score = mean_reciprocal_rank(retrieved_doc_ids, gold_labels)

    total_map += map_score
    total_mrr += mrr_score

    num_claims += 1

avg_map = total_map / num_claims
avg_mrr = total_mrr / num_claims

print(f"Average MAP@{k}: {avg_map}")
print(f"Average MRR@{k}: {avg_mrr}")