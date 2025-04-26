from sentence_transformers import SentenceTransformer
import numpy as np
import json

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_text(self, text):
        return self.model.encode(text)
    
    def create_embeddings(self, data):
        embeddings = {}
        for restaurant in data:
            text = self._prepare_text(restaurant)
            embeddings[restaurant['name']] = {
                'embedding': self.embed_text(text).tolist(),
                'metadata': restaurant
            }
        return embeddings
    
    def _prepare_text(self, restaurant):
        return f"""
        {restaurant['name']}
        Menu: {', '.join([item['name'] for item in restaurant.get('menu', [])]}
        Features: {restaurant.get('features', '')}
        Contact: {restaurant.get('contact', '')}
        """