import pickle

def load_buckets(filename="output/lsh_buckets.pkl"):
    with open(filename, "rb") as file:
        loaded_buckets = pickle.load(file)
    print(f"Buckets loaded from {filename}")
    for bucket in loaded_buckets:
        print(bucket)
    return loaded_buckets

def recommend_similar_docs(given_doc_id, buckets):
    # Initialize an empty set for recommendations
    recommended_doc_ids = set()

    # Iterate over each bucket
    for bucket, doc_indices in buckets.items():
        # If the given document ID is in the current bucket
        if given_doc_id in doc_indices:
            # Add all document IDs in the same bucket to the recommendations
            recommended_doc_ids.update(doc_indices)
    
    # Remove the given document from recommendations to avoid recommending it to itself
    recommended_doc_ids.discard(given_doc_id)
    
    return list(recommended_doc_ids)




if __name__ == '__main__':
    buckets = load_buckets()
    given_doc_id = 'N53283' 
    recommended_docs = recommend_similar_docs(given_doc_id, buckets)
    print(f"Documents recommended for {given_doc_id}: {recommended_docs}")

