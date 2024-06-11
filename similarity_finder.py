from typing import List
import clip
import numpy as np
import torch



class SimilarityFinder:
    def __init__(self, model_name="ViT-B/32", device="cuda") -> None:
        self.device = device
        self.model, _ = clip.load(model_name, device=device)
    
    def embed_text(self, text: str | List[str]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        tokenized_text = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_text).float()

        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def find_similarity(
        self, embedded_prompt: torch.Tensor, embedded_classes: torch.Tensor
    ) -> np.ndarray:
        similarity = embedded_prompt.cpu().numpy() @ embedded_classes.cpu().numpy().T
        return similarity
