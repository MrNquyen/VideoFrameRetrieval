# AIC2025/models/bert_text.py
import torch
import torch.nn.functional as F
from typing import List
from utils.registry import registry

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = AutoModel = None


class TextEncoderBERT:
    """
    Encoder thống nhất:
      - Ưu tiên SentenceTransformer (khuyến nghị: paraphrase-multilingual-MiniLM-L12-v2)
      - Fallback sang HF AutoModel + mean-pooling nếu không có sentence-transformers
    Mặc định normalize=True để dùng với COSINE/IP.
    """
    def __init__(self, model_name: str = "/data2/npl/ViInfographicCaps/model/paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = None, normalize: bool = True):
        self.model_name = model_name
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        if SentenceTransformer is not None and (
            "sentence-transformers" in model_name
            or "intfloat/" in model_name
            or "bge" in model_name
            or "e5" in model_name
        ):
            self.backend = "st"
            self.m = SentenceTransformer(model_name, device=self.device)
            self.dim = self.m.get_sentence_embedding_dimension()
        else:
            assert AutoTokenizer is not None and AutoModel is not None, \
                "Hãy cài sentence-transformers hoặc transformers trước."
            self.backend = "hf"
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.m = AutoModel.from_pretrained(model_name).to(self.device).eval()
            self.dim = self.m.config.hidden_size

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if self.backend == "st":
            with torch.no_grad():
                emb = self.m.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize,
                    device=self.device,
                )
            return emb  # [B, D]
        else:
            # Fallback HF: mean-pooling + optional L2-normalize
            with torch.no_grad():
                b = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                out = self.m(**b).last_hidden_state        # [B, L, H]
                mask = b.attention_mask.unsqueeze(-1)      # [B, L, 1]
                mean = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                if self.normalize:
                    mean = F.normalize(mean, p=2, dim=1)
            return mean