import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(dataset_name, device):
    processed_path = f'./ID_generation/preprocessing/processed'
    embeddings_path = os.path.join(processed_path, f'{dataset_name}_embeddings.npy')
    id2meta_path = os.path.join(processed_path, f'{dataset_name}_id2meta.json')

    if not os.path.exists(embeddings_path):
        print("Embeddings not found, generating embeddings...")
        
        # Load item to text mapping
        with open(id2meta_path, 'r') as f:
            item_2_text = json.load(f)
        
        # Initialize the model
        text_embedding_model = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
        
        # Convert keys to integers and sort by key
        item_id_2_text = {int(k): v for k, v in item_2_text.items()}
        sorted_text = [value for key, value in sorted(item_id_2_text.items())]
        
        # Generate embeddings
        embeddings = text_embedding_model.encode(sorted_text, convert_to_numpy=True, batch_size=512, show_progress_bar=True)
        
        # Save embeddings as .npy
        np.save(embeddings_path, embeddings)
    else:
        print("Embeddings already exist. Loading embeddings...")

    # Load embeddings
    embeddings = np.load(embeddings_path)

    return embeddings

if __name__ == "__main__":
    dataset_name = 'Beauty'  # Replace with your dataset name
    device = 'cuda'  # or 'cpu' for CPU
    
    embeddings = generate_embeddings(dataset_name, device)
    # You can now use the embeddings variable for further processing
