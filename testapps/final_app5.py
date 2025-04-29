import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple

# ─── 1) Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Veggie Detector & Recipe Assistant",
    page_icon="🥗",
    layout="wide",
)


# ─── 2) Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


model, processor = load_clip()

# ─── 3) Labels & Detection Logic ───────────────────────────────────
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
    "scallion",
    "chili",
    "snow pea",
    "baby corn",
]


def detect_vegetables(
    image: Image.Image, labels: List[str], top_k: int = 5, threshold: float = 0.01
) -> List[Tuple[str, float]]:
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_probs, top_idx = probs.topk(top_k, dim=1)

    detections: List[Tuple[str, float]] = []
    for rank, idx in enumerate(top_idx[0]):
        score = top_probs[0, rank].item()
        if score >= threshold:
            detections.append((labels[idx], score))
    return detections


# ─── 4) Sidebar Inputs ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Configuration")

    # 4.1 User Type
    user_type = st.selectbox(
        "User Type",
        ("Free", "Paid"),
        help="Free users can only provide ingredients. Paid users get full recipe customization.",
    )

    # 4.2 Step 1: Ingredients
    st.subheader("Step 1: Ingredients")
    uploaded = st.file_uploader("📷 Upload Image", type=["jpg", "png"])
    typed_ingredients = st.text_area(
        "✏️ Or type ingredients (comma separated)",
        placeholder="e.g. tomato, onion, spinach",
        height=80,
    )

    # 4.3 If image uploaded, ask top_k
    if uploaded:
        top_k = st.number_input(
            "Number of ingredients in the image:",
            min_value=1,
            max_value=len(candidate_labels),
            value=5,
            step=1,
        )
    else:
        top_k = None

    # 4.4 Paid-only advanced options
    if user_type == "Paid":
        with st.expander("Step 2: Advanced Preferences"):
            spice_level = st.slider(
                "🌶️ Spice Level", 0, 10, 5, help="0 = Mild, 10 = Very Spicy"
            )
            recipe_cuisine = st.selectbox(
                "🍽️ Cuisine",
                [
                    "Indian",
                    "Mexican",
                    "Italian",
                    "Chinese",
                    "Thai",
                    "American",
                    "Middle Eastern",
                    "Mediterranean",
                ],
            )
            cook_level = st.selectbox(
                "⏱️ Difficulty",
                ["Easy (10–15 min)", "Medium (15–20 min)", "Hard (20–25 min)"],
            )
            health_goal = st.selectbox(
                "❤️ Health Goal",
                [
                    "General",
                    "Heart Health",
                    "Diabetes-Friendly",
                    "Fatty Liver",
                    "Weight Loss",
                ],
            )

    st.markdown("---")
    run_btn = st.button("🚀 Generate Recipe Plan")

# ─── 5) Main Layout ────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Preview")
    if uploaded:
        st.image(uploaded, use_column_width=True)
    else:
        st.info("Upload an image or type ingredients to begin.")

with col2:
    st.subheader("📝 Results")
    if not (uploaded or typed_ingredients):
        st.write("Waiting for ingredients…")
    elif run_btn:
        # 5.1 Gather ingredients
        final_ings = []
        if uploaded and top_k:
            img = Image.open(uploaded).convert("RGB")
            detected = detect_vegetables(img, candidate_labels, top_k)
            final_ings += [name for name, _ in detected]
        if typed_ingredients:
            final_ings += [ing.strip().lower() for ing in typed_ingredients.split(",")]

        final_ings = sorted(set(final_ings))
        if final_ings:
            st.success("✅ Ingredients Found:")
            for ing in final_ings:
                st.write(f"- {ing.title()}")
        else:
            st.error("❌ No ingredients detected. Try again!")

        # 5.2 Show advanced preferences for paid users
        if user_type == "Paid" and final_ings:
            st.markdown("---")
            st.subheader("🔧 Your Preferences")
            st.write(f"- Spice Level: **{spice_level}/10**")
            st.write(f"- Cuisine: **{recipe_cuisine}**")
            st.write(f"- Difficulty: **{cook_level}**")
            st.write(f"- Health Goal: **{health_goal}**")
