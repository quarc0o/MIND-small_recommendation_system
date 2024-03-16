import configparser 
import sys 
from pathlib import Path 
import os  
import time 
import numpy as np 
import random 
from itertools import combinations 
import pandas as pd
import numpy as np
import random

document_list = dict()

news_df = pd.read_csv('data/training/news.tsv', delimiter='\t', header=None)


for index, row in news_df.iterrows():
    document_id = row[0]  
    title = row[3] if not pd.isna(row[3]) else ""
    abstract = row[4] if not pd.isna(row[4]) else ""
    if abstract == None or title == None:
        continue  
    document_list[document_id] = title + " " + abstract

def k_shingles():
    k = 5
    docs_shingles = []  
    doc_ids = [] 
    for doc_id, doc_content in document_list.items():
        words = doc_content.split()
        shingles = [' '.join(words[i:i+k]) for i in range(len(words) - k + 1)]
        docs_shingles.append(set(shingles)) 
        doc_ids.append(doc_id) 
    return docs_shingles, doc_ids

def signature_matrix(k_shingles_list):
    signature_matrix = []
    flattened_shingles = [item for sublist in k_shingles_list for item in sublist]
    unique_words = list(set(flattened_shingles))  
    signature_matrix = np.zeros((len(k_shingles_list), len(unique_words)))

    for i, shingle_set in enumerate(k_shingles_list):
        for j, word in enumerate(unique_words):
            if word in shingle_set:
                signature_matrix[i, j] = 1
    return signature_matrix



def minhash_signature(docs_shingles, num_hash_functions=100):
    num_docs = len(docs_shingles)
    max_shingle_id = 2**32-1 
    hash_funcs = [lambda x, a=a, b=b: (a * x + b) % max_shingle_id for a, b in 
                  [(random.randint(0, max_shingle_id), random.randint(0, max_shingle_id)) for _ in range(num_hash_functions)]]

    signatures = np.full((num_hash_functions, num_docs), np.inf)

    for doc_idx, shingles in enumerate(docs_shingles):
        for shingle in shingles:
            for i, hash_func in enumerate(hash_funcs):
                hashed_value = hash_func(hash(shingle))  # Simple hash of shingle
                if hashed_value < signatures[i, doc_idx]:
                    signatures[i, doc_idx] = hashed_value

    return signatures

def lsh(signatures, num_bands, num_rows):
    """
    Implements Locality-Sensitive Hashing (LSH) to group similar documents into buckets.

    :param signatures: The MinHash signature matrix.
    :param num_bands: Number of bands to split the signatures into.
    :param num_rows: Number of rows in each band.
    :return: A dictionary of buckets, with each bucket containing document indices considered similar.
    """
    buckets = {}
    band_size = num_rows
    for band in range(num_bands):
        start_row = band * band_size
        end_row = start_row + band_size
        for doc_idx in range(signatures.shape[1]):
            # Create a hashable representation of the band
            band_signature = tuple(signatures[start_row:end_row, doc_idx])
            bucket_key = hash(band_signature)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(doc_idx)
    return buckets

def approximate_jaccard(sig1, sig2):
    """Approximate Jaccard similarity between two MinHash signatures."""
    assert len(sig1) == len(sig2), "Signatures must be the same length"
    return np.mean(sig1 == sig2)

def get_similar_docs(buckets, doc_ids):
    similarity_threshold = 0.6 
    similar_docs = []

    for bucket, doc_indices in buckets.items():
        # Check all pairs in each bucket
        for i in range(len(doc_indices)):
            for j in range(i + 1, len(doc_indices)):
                sim = approximate_jaccard(signature_matrix[:, doc_indices[i]], signature_matrix[:, doc_indices[j]])
                if sim >= similarity_threshold and sim < 1.00:
                    similar_docs.append((doc_ids[doc_indices[i]], doc_ids[doc_indices[j]], sim))
    return similar_docs



if __name__ == '__main__':
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles, doc_ids = k_shingles()  # Unpack the returned doc_ids too
    t5 = time.time()
    print(f"Representing documents with k-shingles took {t5 - t4} sec\n")
    
    signature_matrix = minhash_signature(all_docs_k_shingles)
    print("MinHash signatures generated.")

    num_bands = 20
    rows_per_band = 5  
    buckets = lsh(signature_matrix, num_bands, rows_per_band)
    print("LSH buckets created.")
    
    similar_docs = get_similar_docs(buckets, doc_ids)
    print(f"Found {len(similar_docs)} document pairs with similarity >= {0.6}")
    for doc1, doc2, sim in similar_docs:
        print(f"Doc {doc1} and Doc {doc2} have similarity {sim:.2f}")
