import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple

st.set_page_config(
    page_title="Veggie Detector & Recipe Assistant",
    page_icon="🥗",
    layout="wide",  # optional: wide, centered, etc.
)


# --- Load CLIP model ---
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


model, processor = load_clip()

# --- Vegetable Labels ---
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
    "eggplant",
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


# --- Vegetable Detection Logic ---
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
        label = labels[idx]
        if score >= threshold:
            detections.append((label, score))
    return detections


# --- Streamlit App ---

st.title("🥗 Smart Vegetable Detector & Recipe Builder")

# --- User Type Selection ---
user_type = st.radio(
    "Select your User Type:",
    ["Free User", "Paid User"],
    index=0,
    help="Paid users can access advanced cooking and health preferences.",
)

# --- Upload Image OR Type Ingredients Manually ---
st.header("Step 1: Provide Your Ingredients")

uploaded = st.file_uploader(
    "Upload an image of ingredients", type=["jpg", "jpeg", "png"]
)
typed_ingredients = st.text_area(
    "Or, type ingredients manually (comma separated):",
    placeholder="Example: tomato, onion, spinach",
)

# --- Validation: must fill one ---
if not uploaded and not typed_ingredients:
    st.warning("⚠️ Please either upload an image OR manually type ingredients.")

else:
    # --- If paid user, show advanced options ---
    if user_type == "Paid User":
        st.header("Step 2: Customize Your Recipe")

        # Spice Level Slider
        spice_level = st.slider(
            "Select Spice Level (0 = Mild, 10 = Very Spicy):",
            min_value=0,
            max_value=10,
            value=5,
        )

        # Cuisine Type
        recipe_cuisine = st.selectbox(
            "Select Recipe Cuisine:",
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

        # Cooking Difficulty
        cook_level = st.selectbox(
            "Select Cooking Difficulty:",
            ["Easy (10-15 minutes)", "Medium (15-20 minutes)", "Hard (20-25 minutes)"],
        )

        # Health Goal
        health_goal = st.selectbox(
            "Select Your Health Goal:",
            [
                "General Health",
                "Heart Health",
                "Diabetes-Friendly",
                "Fatty Liver Recovery",
                "Weight Loss",
            ],
        )

    st.header("Step 3: Detect Ingredients and Build Recipe")
    if st.button("Generate Recipe Plan"):
        final_ingredients = []

        if uploaded:
            with st.spinner("Detecting ingredients from image..."):
                img = Image.open(uploaded).convert("RGB")
                detected = detect_vegetables(img, candidate_labels, top_k=5)
                final_ingredients = [name for name, score in detected]

        if typed_ingredients:
            manual_ingredients = [
                i.strip().lower() for i in typed_ingredients.split(",") if i.strip()
            ]
            final_ingredients.extend(manual_ingredients)

        # Deduplicate
        final_ingredients = list(set(final_ingredients))

        if final_ingredients:
            st.success("✅ Ingredients Found:")
            for ing in final_ingredients:
                st.write(f"- {ing.title()}")

            if user_type == "Paid User":
                st.info(f"🍴 **Recipe Preferences:**")
                st.write(f"- **Spice Level:** {spice_level}/10")
                st.write(f"- **Cuisine:** {recipe_cuisine}")
                st.write(f"- **Cooking Difficulty:** {cook_level}")
                st.write(f"- **Health Goal:** {health_goal}")
        else:
            st.error("❌ No ingredients detected or typed properly. Try again!")
