import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple


# 1. Load and cache CLIP model + processor
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


model, processor = load_clip()

# 2. Your vegetable labels
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
    "cauliflower",
    "zucchini",
    # "eggplant",
    "pepper",
    "peas",
    "corn",
    "radish",
    "celery",
    "garlic",
    "ginger",
    "mushroom",
    "green bean",
    "asparagus",
    "spring onion",
    "chili",
    "snow pea",
    "baby corn",
    "cherry tomato",
    "scallion",
    "tenderstem broccoli",
]


# 3. Detection function
def detect_vegetables(
    image: Image.Image, labels: List[str], top_k: int = 10, threshold: float = 0.01
) -> List[Tuple[str, float]]:
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_probs, top_idx = probs.topk(top_k, dim=1)

    detections: List[Tuple[str, float]] = []
    for rank, idx in enumerate(top_idx[0]):
        score = top_probs[0, rank].item()
        label = labels[idx]
        if score >= threshold:
            detections.append((label, score))
    return detections


# 4. Streamlit UI
st.title("🍅🥕 Recipe Recommendation System")
st.write(
    "Upload a vegetable image and specify how many ingredients you think it contains."
)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Ask user for estimated number of ingredients
    top_k = st.number_input(
        "How many ingredients are in the image?",
        min_value=1,
        max_value=len(candidate_labels),
        value=5,
        step=1,
    )

    if st.button("Detect Vegetables"):
        with st.spinner("Running CLIP inference..."):
            results = detect_vegetables(img, candidate_labels, top_k)

        if results:
            st.subheader("Detected Vegetables:")
            for name, score in results:
                # st.write(f"- **{name}**  ({score:.2f})")
                st.write(f"- **{name}** ")
        else:
            st.warning("No vegetables detected above threshold.")
