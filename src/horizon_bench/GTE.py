import logging
from typing import Optional
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class ModelManager:
    """
    Manages the loading and initialization of transformer models and their tokenizers.
    """
    MODEL_MAPPING = {
        "Alibaba-NLP/gte-base-en-v1.5": AutoModel,
        "Alibaba-NLP/gte-large-en-v1.5": AutoModel,
    }
    DEFAULT_MODEL = "Alibaba-NLP/gte-large-en-v1.5"

    def __init__(self):
        """
        Initializes the ModelManager with no loaded model or tokenizer.
        """
        self._model = None
        self._tokenizer = None

    @property
    def model(self) -> Optional[AutoModel]:
        """
        Retrieves the loaded transformer model.

        Returns:
            Optional[AutoModel]: The loaded transformer model if initialized, else None.
        """
        if not self._model:
            logging.warning("Model not initialized yet. Returning None.")
            return None
        return self._model

    def initialize_model(self, model_name: str, **model_kwargs) -> Optional[AutoModel]:
        """
        Loads and initializes the specified transformer model.

        Args:
            model_name (str): The name of the model to load.
            **model_kwargs: Additional keyword arguments for the model initialization.

        Returns:
            Optional[AutoModel]: The loaded transformer model instance, or None if the model name is invalid.
        """
        if model_name not in self.MODEL_MAPPING:
            logging.error("Model name not found in the mapping. Returning None.")
            return None
        self._model = self.MODEL_MAPPING[model_name].from_pretrained(model_name, **model_kwargs)
        return self._model

    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
        """
        Retrieves the loaded tokenizer.

        Returns:
            Optional[AutoTokenizer]: The loaded tokenizer if initialized, else None.
        """
        if not self._tokenizer:
            logging.warning("Tokenizer not initialized yet. Returning None.")
            return None
        return self._tokenizer

    def initialize_tokenizer(self, model_name: str, **tokenizer_kwargs) -> Optional[AutoTokenizer]:
        """
        Loads and initializes the tokenizer for the specified model.

        Args:
            model_name (str): The name of the model whose tokenizer to load.
            **tokenizer_kwargs: Additional keyword arguments for the tokenizer initialization.

        Returns:
            Optional[AutoTokenizer]: The loaded tokenizer instance.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        return self._tokenizer


class GTEWorker:
    """
    LLM Worker class to manage transformer models, embeddings, and related resources.
    """
    def __init__(self, device: str = "cuda"):
        """
        Initializes the GTEWorker with a ModelManager and sets the computation device.

        Args:
            device (str, optional): The device to run the model on (e.g., "cuda" or "cpu"). Defaults to "cuda".
        """
        self.model_manager = ModelManager()
        self.device = device
        self.pca_components = None
        self.mean_embedding = None
        self.std_embedding = None

    def initialize_model(self, model_name: str, **model_kwargs) -> Optional[AutoModel]:
        """
        Initializes the transformer model and moves it to the specified device.

        Args:
            model_name (str): The name of the model to initialize.
            **model_kwargs: Additional keyword arguments for the model initialization.

        Returns:
            Optional[AutoModel]: The initialized and device-moved transformer model, or None if initialization failed.
        """
        model = self.model_manager.initialize_model(model_name, **model_kwargs)
        if model:
            return model.to(self.device)
        return None

    def initialize_tokenizer(self, model_name: str, **tokenizer_kwargs) -> Optional[AutoTokenizer]:
        """
        Initializes the tokenizer for the transformer model.

        Args:
            model_name (str): The name of the model whose tokenizer to initialize.
            **tokenizer_kwargs: Additional keyword arguments for the tokenizer initialization.

        Returns:
            Optional[AutoTokenizer]: The initialized tokenizer, or None if initialization failed.
        """
        return self.model_manager.initialize_tokenizer(model_name, **tokenizer_kwargs)

    def prepare_input_gte(self, title : str, abstract : str) -> str:
        """
        Any preprocessing required for the input before passing it to the model.
        For now, it adds a [SEP] token between the title and abstract and tokenize the input.

        Args:
            title (str): Title of the paper.
            abstract (str): Abstract of the paper.

        Returns:
            str: Preprocessed input.
        """
        return title + '.' + self.model_manager.tokenizer.sep_token + abstract