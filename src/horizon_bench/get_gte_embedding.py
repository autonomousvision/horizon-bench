import torch
from typing import List
import numpy as np
from horizon_bench.GTE import GTEWorker

gte_worker = GTEWorker(device="cuda")

def get_paper_embeddings(titles: str, abstracts: str):
    """
    Get GTE embeddings for the given titles and abstracts.
    
    Args:
        titles (List[str]): A list of paper titles.
        abstracts (List[str]): A list of paper abstracts.

    Returns: 
        List[dict]: A list of dictionaries containing the GTE embeddings for each paper.
    """
    texts = [f"{title}\n{abstract}" for title, abstract in zip(titles, abstracts)]
    return get_gte_embeddings(texts)


def get_gte_embeddings(
        texts : str | List[str],
        batch_size: int = 16
):
    """
    Takes any string or list of strings and returns the GTE embedding(s).
    """
    if isinstance(texts, str):
        texts = [texts]
        
    global gte_worker
    model_name = gte_worker.model_manager.DEFAULT_MODEL
    
    if gte_worker.model_manager.model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gte_worker = GTEWorker(device=device)
        gte_worker.initialize_model(model_name,
                                    torch_dtype=torch.float16,
                                    trust_remote_code=True,
                                    unpad_inputs=True,
                                    use_memory_efficient_attention=True).to(device)
        gte_worker.initialize_tokenizer(model_name)

    model = gte_worker.model_manager.model
    tokenizer = gte_worker.model_manager.tokenizer
    device = gte_worker.device

    gte_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = _get_gte_batch_embeddings(batch_texts, model, tokenizer, device)
        gte_embeddings.extend(batch_embeddings)
    return gte_embeddings


def _get_gte_batch_embeddings(texts, model, tokenizer, device):
    # Get the model embeddings
    batch_dict = tokenizer(texts,
                            max_length=2048, 
                            padding='max_length', 
                            truncation=True, 
                            return_tensors='pt')
        
    batch_dict = batch_dict.to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)
    # Embedding is the class token
    embeddings = outputs.last_hidden_state[:, 0] # [batch_size, sequence_length, hidden_size] -> [batch_size, hidden_size]

    # convert to float16
    embeddings = embeddings.to(torch.float16)

    pinned_memory = torch.empty(embeddings.shape, dtype=embeddings.dtype, pin_memory=True)
    pinned_memory.copy_(embeddings, non_blocking=False)
    embeddings = pinned_memory.cpu().numpy()

    # Check whether there is a inf or nan value in the embeddings and map it to 0
    # print(f"Number of nan in embeddings = {np.sum(np.isnan(embeddings))}")
    # Check number of zero vectors
    # zero_indices = np.where(np.all(embeddings == 0, axis=1))[0]
    # print(f"Number of zero vectors = {len(zero_indices)}")

    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    return embeddings

if __name__ == "__main__":
    titles = [
        "Attention Is All You Need",
    ]
    abstracts = [
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by more than 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.",
    ]

    embeddings = get_paper_embeddings(titles, abstracts)
    print(embeddings)