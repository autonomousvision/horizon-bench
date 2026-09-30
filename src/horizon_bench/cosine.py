import numpy as np


def cosine(embedding_a, embedding_b):
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding_a (np.ndarray): A 1D numpy array representing the first embedding.
        embedding_b (np.ndarray): A 1D numpy array representing the second embedding.
    """
    return np.dot(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))

def get_cosine_similarity(embedding, embedding_list):
    """
    Calculate cosine similarity between a single embedding and a list of embeddings.
    
    Args:
        embedding (np.ndarray): A 1D numpy array representing the embedding.
        embedding_list (List[np.ndarray]): A list of 1D numpy arrays representing the embeddings to compare against.

    Returns:
        List[float]: A list of cosine similarity scores.
    """
    similarities = []
    for emb in embedding_list:
        cos_sim = cosine(embedding, emb)
        similarities.append(float(cos_sim))
    return similarities

if __name__ == "__main__":
    pass