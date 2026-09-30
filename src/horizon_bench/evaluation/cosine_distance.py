import os
import json
from horizon_bench.get_gte_embedding import get_gte_embeddings
from horizon_bench.cosine import cosine
from horizon_bench.evaluation.load_idea import load_idea
from horizon_bench.Config import GROUND_TRUTH_DIR, OUTPUT_DIR


def get_cosine_similarity(text_a:str, text_b:str) -> float:
    """
    Compute the cosine similarity between GTE embeddings of text_a and text_b
    """
    idea_embedding = get_gte_embeddings(text_a)[0]
    gt_embedding = get_gte_embeddings(text_b)[0]
    cosine_similarity = cosine(idea_embedding, gt_embedding)
    return cosine_similarity

if __name__ == "__main__":
    pass