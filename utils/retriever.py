import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def retrieve(self, query, embedder, top_k=3):
        query_embed = embedder.embed_text(query)
        results = []
        
        for name, data in self.embeddings.items():
            sim = cosine_similarity(
                [query_embed],
                [np.array(data['embedding'])]
            )[0][0]
            results.append((sim, data['metadata']))
        
        # Sort by similarity score
        results.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in results[:top_k]]