import math
from typing import List, Dict, Tuple
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from qasper_loader import load_qasper_documents


def _collect_paragraphs(split: str, granularity: str = "paragraph") -> Tuple[List[str], List[Dict]]:
    """
    Load QASPER paragraphs as raw strings and keep their metadata.

    Returns two parallel lists:
    - texts[i]: paragraph content (str)
    - metas[i]: paragraph meta dict with keys like paper_id, para_idx, title
    """
    texts: List[str] = []
    metas: List[Dict] = []
    for item in load_qasper_documents(split=split, granularity=granularity):
        texts.append(item["content"])
        metas.append(item.get("meta", {}))
    return texts, metas


class TfidfRetriever:
    """
    A tiny TF-IDF retriever using scikit-learn style API (fit/score) but implemented
    with sklearn to keep the code simple and consistent with the repo level.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy import
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._similarity = cosine_similarity
        self._doc_matrix = None
        self._metas: List[Dict] = []

    def fit(self, split: str = "train", granularity: str = "paragraph") -> None:
        texts, metas = _collect_paragraphs(split, granularity=granularity)
        self._metas = metas
        self._doc_matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if self._doc_matrix is None:
            raise RuntimeError("Call fit() before retrieve().")
        q_vec = self._vectorizer.transform([query])
        sims = self._similarity(q_vec, self._doc_matrix)[0]
        # Get top_k indices
        top_idx = sims.argsort()[::-1][:top_k]
        outputs: List[Dict] = []
        for i in top_idx:
            meta = self._metas[i]
            outputs.append({
                "content": None,  # content omitted to keep memory small in evaluation; retrieve if needed
                "meta": meta,
                "score": float(sims[i]),
            })
        return outputs


class LsiRetriever:
    """
    LSI/LSA retriever implemented via TruncatedSVD on the TF-IDF matrix.
    """

    def __init__(self, n_components: int = 200, random_state: int = 0):
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy import
        from sklearn.decomposition import TruncatedSVD  # lazy import
        from sklearn.preprocessing import Normalizer  # lazy import
        from sklearn.pipeline import make_pipeline  # lazy import
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._normalizer = Normalizer(copy=False)
        self._pipeline = make_pipeline(self._svd, self._normalizer)
        self._similarity = cosine_similarity
        self._doc_matrix = None
        self._doc_lsi = None
        self._metas: List[Dict] = []

    def fit(self, split: str = "train", granularity: str = "paragraph") -> None:
        texts, metas = _collect_paragraphs(split, granularity=granularity)
        self._metas = metas
        self._doc_matrix = self._vectorizer.fit_transform(texts)
        self._doc_lsi = self._pipeline.fit_transform(self._doc_matrix)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if self._doc_lsi is None:
            raise RuntimeError("Call fit() before retrieve().")
        q_vec = self._vectorizer.transform([query])
        q_lsi = self._pipeline.transform(q_vec)
        sims = self._similarity(q_lsi, self._doc_lsi)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        outputs: List[Dict] = []
        for i in top_idx:
            meta = self._metas[i]
            outputs.append({
                "content": None,
                "meta": meta,
                "score": float(sims[i]),
            })
        return outputs


def build_dpr_retriever(model_name: str = "facebook/dpr-question_encoder-single-nq-base"):
    """
    Construct a Haystack-like in-memory DPR pipeline using sentence transformers DPR models.
    We'll reuse the existing `index_qasper.build_and_index` for documents, then build a
    query encoder and retriever. This function returns a tuple (store, pipeline).
    """
    try:
        from haystack import Pipeline  # type: ignore
        from haystack.components.embedders import SentenceTransformersTextEmbedder  # type: ignore
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever  # type: ignore
        from haystack.document_stores.in_memory import InMemoryDocumentStore  # type: ignore
        from index_qasper import build_and_index
    except Exception as e:
        raise RuntimeError("Haystack is required for DPR baseline: pip install haystack-ai") from e

    # Build document index using the DPR context encoder model
    store: InMemoryDocumentStore = build_and_index(model_name=model_name, split="train")
    text_embedder = SentenceTransformersTextEmbedder(model=model_name)
    retriever = InMemoryEmbeddingRetriever(store)

    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("retriever", retriever)
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")

    if hasattr(text_embedder, "warm_up"):
        text_embedder.warm_up()

    return store, pipe


class SpladeRetriever:
    """
    SPLADE (SParse Lexical And Dense Embeddings) retriever.
    
    SPLADE learns sparse representations where the model predicts importance weights
    for vocabulary terms. Unlike TF-IDF which uses fixed formulas, SPLADE learns
    which terms are important for retrieval through neural training.
    
    This implementation uses a pre-trained SPLADE model from HuggingFace.
    """

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", 
                 batch_size: int = 32, max_workers: int = None):
        """
        Initialize SPLADE retriever with optimized batch processing.
        
        Args:
            model_name: HuggingFace model name for SPLADE
            batch_size: Batch size for GPU processing (optimized for RTX A6000)
            max_workers: Number of parallel workers for CPU tasks
        """
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            import torch
            import numpy as np
        except ImportError as e:
            raise RuntimeError("SPLADE requires transformers and torch: pip install transformers torch") from e
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Use available cores
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained SPLADE model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Enable mixed precision for faster processing on RTX A6000
        if self.device.type == "cuda":
            self.model = self.model.half()  # Use FP16 for 2x speedup
        
        # Storage for document representations
        self._doc_sparse_reps = None
        self._metas: List[Dict] = []
        self._vocab_size = self.tokenizer.vocab_size

    def _encode_text(self, text: str) -> Dict[int, float]:
        """
        Encode text into SPLADE sparse representation.
        
        Returns dict mapping token_id -> importance_weight
        """
        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get logits from masked language model
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Apply ReLU and log(1 + x) as in SPLADE paper
            # This gives us sparse representations with non-negative weights
            sparse_rep = torch.log(1 + torch.relu(logits))
            
            # Max pooling over sequence dimension to get document-level representation
            doc_rep = torch.max(sparse_rep, dim=1)[0].squeeze()  # [vocab_size]
            
            # Convert to sparse dict (only store non-zero weights)
            sparse_dict = {}
            for token_id, weight in enumerate(doc_rep.cpu().numpy()):
                if weight > 0.01:  # Threshold to keep only significant weights
                    sparse_dict[token_id] = float(weight)
            
            return sparse_dict

    def _encode_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Encode a batch of texts into SPLADE sparse representations.
        This is much faster than encoding one by one on GPU.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            List of sparse representations (dicts)
        """
        # Tokenize batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, 
                              truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get logits from masked language model
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Apply ReLU and log(1 + x) as in SPLADE paper
            sparse_rep = torch.log(1 + torch.relu(logits))
            
            # Max pooling over sequence dimension to get document-level representation
            doc_reps = torch.max(sparse_rep, dim=1)[0]  # [batch_size, vocab_size]
            
            # Convert to sparse dicts (only store non-zero weights)
            sparse_dicts = []
            for doc_rep in doc_reps:
                sparse_dict = {}
                for token_id, weight in enumerate(doc_rep.cpu().numpy()):
                    if weight > 0.01:  # Threshold to keep only significant weights
                        sparse_dict[token_id] = float(weight)
                sparse_dicts.append(sparse_dict)
            
            return sparse_dicts

    def fit(self, split: str = "train", granularity: str = "paragraph") -> None:
        """Fit SPLADE on the document collection using optimized batch processing."""
        print(f"Building SPLADE representations for {split} split...")
        print(f"Using batch size: {self.batch_size}, GPU: {self.device}")
        texts, metas = _collect_paragraphs(split, granularity=granularity)
        self._metas = metas
        
        # Process documents in batches for much faster GPU utilization
        doc_reps = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_texts)} documents)...")
            
            # Encode batch on GPU - much faster than individual encoding
            batch_reps = self._encode_batch(batch_texts)
            doc_reps.extend(batch_reps)
        
        self._doc_sparse_reps = doc_reps
        print(f"SPLADE fitting completed for {len(texts)} documents using {total_batches} batches")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents using SPLADE sparse matching."""
        if self._doc_sparse_reps is None:
            raise RuntimeError("Call fit() before retrieve().")
        
        # Encode query into sparse representation
        query_rep = self._encode_text(query)
        
        # Compute scores against all documents using sparse dot product
        scores = []
        for doc_rep in self._doc_sparse_reps:
            # Sparse dot product between query and document
            score = 0.0
            for token_id, query_weight in query_rep.items():
                if token_id in doc_rep:
                    score += query_weight * doc_rep[token_id]
            scores.append(score)
        
        # Get top-k results
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        outputs = []
        for idx in top_indices:
            meta = self._metas[idx]
            outputs.append({
                "content": None,  # content omitted for memory efficiency
                "meta": meta,
                "score": float(scores[idx])
            })
        
        return outputs




def build_splade_retriever(model_name: str = "naver/splade-cocondenser-ensembledistil") -> SpladeRetriever:
    """
    Build a SPLADE retriever with the specified model.
    
    Args:
        model_name: HuggingFace model name for SPLADE
        
    Returns:
        Configured SpladeRetriever instance
    """
    return SpladeRetriever(model_name=model_name)




