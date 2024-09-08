import pickle

# Specify the file path where your embeddings were saved
input_file = "scifact_evidence_embeddings.pkl"

# Step 1: Load the pickled embeddings from disk
with open(input_file, "rb") as f:
    embeddings = pickle.load(f)

# Now, `embeddings` is a dictionary containing your evidence documents and their corresponding embeddings
# Example: Accessing an embedding
for doc, embedding in embeddings.items():
    doc_id, abstract = doc
    print("------")
    print(f"Document ID: {doc_id}")
    print(f"Abstract: {abstract}")
    print(f"Embedding: {embedding}\n")