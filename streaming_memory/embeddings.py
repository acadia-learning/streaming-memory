"""
Local embedding service using Qwen3-Embedding-8B.

This provides fast, local embeddings without needing external API calls.
"""

from typing import Optional

import numpy as np
import torch


class LocalEmbedder:
    """
    Local embedding model using Qwen3-Embedding-8B.

    Much faster than API calls since it runs locally with no network latency.
    Supports batch processing for efficiency.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        cache_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize the local embedder.

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-small-en-v1.5)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            cache_embeddings: Whether to cache embeddings for repeated texts
            batch_size: Batch size for batch encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        print(f"Loading embedding model {model_name} on {device}...")

        # Use sentence-transformers for better compatibility
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
            )
            self._use_sentence_transformers = True
            print(f"✓ Model loaded using sentence-transformers on {device}")
        except ImportError:
            # Fallback to transformers
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                trust_remote_code=True,
            ).to(device)
            self.model.eval()
            self._use_sentence_transformers = False
            print(f"✓ Model loaded using transformers on {device}")

        # Cache for repeated embeddings
        self.cache_embeddings = cache_embeddings
        self.embed_cache: dict[str, np.ndarray] = {}

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        if self.cache_embeddings and text in self.embed_cache:
            return self.embed_cache[text]

        if self._use_sentence_transformers:
            # Use sentence-transformers
            embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        else:
            # Use transformers directly
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Get embeddings
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])

            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Convert to numpy
            embedding = embeddings[0].cpu().numpy()

        # Cache if enabled
        if self.cache_embeddings:
            self.embed_cache[text] = embedding

        return embedding

    @torch.no_grad()
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed a batch of texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors as numpy arrays
        """
        # Check cache first
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            if self.cache_embeddings and text in self.embed_cache:
                results[i] = self.embed_cache[text]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # If all cached, return early
        if not texts_to_embed:
            return results

        if self._use_sentence_transformers:
            # Use sentence-transformers batch encoding
            all_embeddings = self.model.encode(
                texts_to_embed,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            # Process in batches manually
            all_embeddings = []
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch = texts_to_embed[i:i + self.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Get embeddings
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs["attention_mask"])

                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to numpy
                batch_embeddings = embeddings.cpu().numpy()
                all_embeddings.extend(batch_embeddings)

        # Fill in results and cache
        for idx, embedding in zip(indices_to_embed, all_embeddings):
            results[idx] = embedding
            if self.cache_embeddings:
                self.embed_cache[texts[idx]] = embedding

        return results

    def __call__(self, text: str) -> np.ndarray:
        """Allow using embedder as a callable."""
        return self.embed(text)

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embed_cache.clear()

    def get_cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self.embed_cache)


def create_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: Optional[str] = None,
    cache_embeddings: bool = True,
    batch_size: int = 32,
) -> LocalEmbedder:
    """
    Factory function to create a local embedder.

    Args:
        model_name: HuggingFace model name
        device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
        cache_embeddings: Whether to cache embeddings
        batch_size: Batch size for batch encoding

    Returns:
        Configured LocalEmbedder instance
    """
    return LocalEmbedder(
        model_name=model_name,
        device=device,
        cache_embeddings=cache_embeddings,
        batch_size=batch_size,
    )

