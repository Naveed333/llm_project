from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple


# Load model & processor once
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


model, processor = load_clip_model()

# Candidate labels
candidate_labels: List[str] = [
    "tomato",
    "potato",
    "onion",
    "carrot",
    "cucumber",
    "spinach",
    "lettuce",
    "cabbage",
    "broccoli",
    # "cauliflower",
    "zucchini",
    "pepper",
    "peas",
    "corn",
    "radish",
    "celery",
    "garlic",
    "ginger",
    "lemon",
    "Green Chilli",
    # "mushroom",
]


# Detection function
def detect_vegetables(
    image: Image.Image, labels: List[str], top_k: int = 5, threshold: float = 0.01
) -> List[Tuple[str, float]]:
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_probs, top_idx = probs.topk(top_k, dim=1)
    results = []
    for i, idx in enumerate(top_idx[0]):
        score = top_probs[0, i].item()
        if score >= threshold:
            results.append((labels[idx], score))
    return results
