import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple

# ─── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="Veggie Detector & Recipe Assistant", page_icon="🥗", layout="wide"
)

# ─── McDonalds-style Theme ────────────────────────────────────────
st.markdown(
    """
    <style>
      .css-18e3th9 { font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', sans-serif; }
      .block-container { padding: 2rem 3rem; background-color: #FFFFFF; }
      /* Sidebar background */
      .css-1d391kg { background-color: #DA291C; }
      /* Headings in primary color */
      h1, h2, h3, h4, h5, h6 { color: #DA291C; }
      /* Buttons styling */
      .stButton>button {
        border-radius: 24px;
        background-color: #FFC72C;
        color: #DA291C;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
      }
      /* Slider handles and active color */
      .css-16gz6vm .css-1lsmgbg {  /* slider track */
        background: #FFC72C;
      }
      .css-16gz6vm .css-1lsmgbg .css-1sw0di0 { /* slider handle */
        background: #DA291C;
      }
      /* Checkbox and radio selected color */
      .stCheckbox input:checked + label:before,
      .stRadio input:checked + label:before {
        background-color: #DA291C;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Load CLIP model ──────────────────────────────────────────────
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


model, processor = load_clip()

# ─── Labels & Detection ───────────────────────────────────────────
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
    "pepper",
    "peas",
    "corn",
    "radish",
    "celery",
    "garlic",
    "ginger",
    "mushroom",
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


# ─── Session State for Tab Control ─────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "ImagePreview"

# ─── Sidebar Inputs ───────────────────────────────────────────────
st.sidebar.header("🛠️ Configuration")
user_type = st.sidebar.radio(
    "User Type",
    options=["Free", "Paid"],
    index=0,
    help="Free: basic input. Paid: unlocks advanced options.",
)

st.sidebar.subheader("1️⃣ Ingredients Input")
uploaded = st.sidebar.file_uploader(
    "Upload Image of Ingredients",
    type=["jpg", "png"],
    help="Optional photo of your veggies.",
)
manual_input = st.sidebar.text_input("Or Type Ingredients (comma-separated)", "")

# If image provided, ask number of items
top_k = None
if uploaded:
    top_k = st.sidebar.slider(
        "How many items to detect?",
        1,
        len(candidate_labels),
        5,
        help="Select how many vegetables to identify from photo.",
    )

# Advanced settings for Paid users
st.sidebar.markdown("---")
st.sidebar.subheader("2️⃣ Advanced Settings")
spice_level = st.sidebar.slider(
    "Spice Level",
    0,
    10,
    5,
    disabled=(user_type == "Free"),
    help="0=Mild, 10=Very Spicy",
)
recipe_cuisine = st.sidebar.selectbox(
    "Cuisine Type",
    ["Indian", "Mexican", "Italian", "Chinese", "Thai", "American", "Mediterranean"],
    disabled=(user_type == "Free"),
)
cook_time = st.sidebar.selectbox(
    "Difficulty & Time",
    ["Easy (10-15 min)", "Medium (15-20 min)", "Hard (20-25 min)"],
    disabled=(user_type == "Free"),
)
health_goals = st.sidebar.multiselect(
    "Health Goals",
    [
        "General Health",
        "Heart Health",
        "Diabetes-Friendly",
        "Fatty Liver",
        "Weight Loss",
    ],
    default=["General Health"],
    disabled=(user_type == "Free"),
)

# Generate Button
if st.sidebar.button("🚀 Generate Recipe Plan"):
    st.session_state.active_tab = "Results"

# ─── Main Area ─────────────────────────────────────────────────────
if st.session_state.active_tab == "ImagePreview":
    st.header("📸 Image Preview & Input Summary")
    cols = st.columns(2)
    with cols[0]:
        if uploaded:
            st.image(uploaded, use_column_width=True)
        else:
            st.info("Upload an image via the sidebar.")
    with cols[1]:
        if manual_input:
            st.markdown(f"**Typed Ingredients:** {manual_input}")
elif st.session_state.active_tab == "Results":
    st.header("📝 Detected Ingredients & Recipe Details")
    # Compile ingredients
    final_ings: List[str] = []
    if uploaded and top_k:
        img = Image.open(uploaded).convert("RGB")
        detected = detect_vegetables(img, candidate_labels, top_k)
        final_ings += [name for name, _ in detected]
    if manual_input:
        final_ings += [i.strip().lower() for i in manual_input.split(",") if i.strip()]
    final_ings = sorted(set(final_ings))

    # Display ingredients
    if final_ings:
        st.success("**Ingredients Identified:**")
        for ing in final_ings:
            st.write(f"• {ing.title()}")
    else:
        st.error("No ingredients found. Please try again.")

    # Show preferences for Paid users
    if user_type == "Paid" and final_ings:
        st.markdown("---")
        st.subheader("👨‍🍳 Your Preferences")
        st.write(f"**Spice Level:** {spice_level}/10")
        st.write(f"**Cuisine:** {recipe_cuisine}")
        st.write(f"**Cooking Time:** {cook_time}")
        st.write(f"**Health Goals:** {', '.join(health_goals)}")
        st.markdown("---")
        st.text_area(
            "Generated Recipe:",
            "Your detailed recipe will appear here based on your inputs.",
            height=200,
        )

# Back button on results
if st.session_state.active_tab == "Results":
    if st.button("🔄 Back to Input"):
        st.session_state.active_tab = "ImagePreview"
