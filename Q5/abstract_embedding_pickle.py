import pickle
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# Step 1: Download the SciFact dataset
dataset = load_dataset("allenai/scifact", "corpus")

# Step 2: Extract all evidence documents
evidence_docs = []
for item in dataset['train']:
    evidence_docs.append(
        (
            item['doc_id'],  # "doc_id"
            " ".join(item['abstract'])  # "abstract"
        )
    )

# Step 3: Generate OpenAI embeddings for the evidence documents
# OpenAI API allows you to generate embeddings using models
# If your dataset is large, you might want to do this in batches.

client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


embeddings = {doc: get_embedding(doc[1]) for doc in tqdm(evidence_docs)}

# Step 4: Save/cache the embeddings on disk
output_file = "scifact_evidence_embeddings.pkl"
with open(output_file, "wb") as f:
    pickle.dump(embeddings, f)

print(f"Embeddings have been saved to {output_file}.")
