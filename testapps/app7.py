import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
import sqlite3
import hashlib

# ─── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="Veggie Detector & Recipe Assistant", page_icon="🥗", layout="wide"
)

# ─── McDonald’s-style Theme ───────────────────────────────────────
st.markdown(
    """
    <style>
      .css-18e3th9 { font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', sans-serif; }
      .block-container { padding: 2rem 3rem; background-color: #FFFFFF; }
      .css-1d391kg { background-color: #DA291C; }
      h1,h2,h3,h4{ color:#DA291C; }
      .stButton>button { border-radius:24px; background-color:#FFC72C; color:#DA291C; font-weight:bold; }
      .css-16gz6vm .css-1lsmgbg { background:#FFC72C; }
      .css-16gz6vm .css-1lsmgbg .css-1sw0di0 { background:#DA291C; }
      .stCheckbox input:checked+label:before, .stRadio input:checked+label:before { background-color:#DA291C; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Database Setup ───────────────────────────────────────────────
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute(
    """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  username TEXT UNIQUE,
  password_hash TEXT,
  subscription TEXT
)
"""
)
c.execute(
    """
CREATE TABLE IF NOT EXISTS preferences (
  user_id INTEGER PRIMARY KEY,
  spice_level INTEGER,
  cuisine TEXT,
  cook_time TEXT,
  health_goals TEXT,
  FOREIGN KEY(user_id) REFERENCES users(id)
)
"""
)
conn.commit()

# ─── Authentication ───────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.user_id = None


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


with st.sidebar.expander("🔒 Account", expanded=True):
    auth_action = st.radio("Action", ["Login", "Register"], index=0)
    username = st.text_input("Username", key="auth_user")
    password = st.text_input("Password", type="password", key="auth_pw")
    if st.button(auth_action):
        if auth_action == "Register":
            pw_hash = hash_pw(password)
            try:
                c.execute(
                    "INSERT INTO users(username,password_hash,subscription) VALUES(?,?,?)",
                    (username, pw_hash, "Free"),
                )
                conn.commit()
                st.success("Registration successful! Please log in.")
            except sqlite3.IntegrityError:
                st.error("Username already exists.")
        else:  # Login
            c.execute(
                "SELECT id,password_hash,subscription FROM users WHERE username=?",
                (username,),
            )
            row = c.fetchone()
            if row and row[1] == hash_pw(password):
                st.success(f"Logged in as {username}")
                st.session_state.user = username
                st.session_state.user_id = row[0]
                st.session_state.subscription = row[2]
            else:
                st.error("Invalid credentials.")


# ─── Load Model ────────────────────────────────────────────────────
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


model, processor = load_clip()

# ─── Detection Logic ──────────────────────────────────────────────
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
    detections = []
    for i, idx in enumerate(top_idx[0]):
        score = top_probs[0, i].item()
        if score >= threshold:
            detections.append((labels[idx], score))
    return detections


# ─── Session State for Tabs ───────────────────────────────────────
if "active" not in st.session_state:
    st.session_state.active = "Input"

# ─── Sidebar: Inputs & Subscription ───────────────────────────────
st.sidebar.header("🛠️ Configuration")


# Subscription management
def upgrade():
    c.execute(
        "UPDATE users SET subscription='Paid' WHERE id=?", (st.session_state.user_id,)
    )
    conn.commit()
    st.session_state.subscription = "Paid"


if st.session_state.user:
    st.sidebar.markdown(f"**Subscription:** {st.session_state.subscription}")
    if st.session_state.subscription == "Free":
        st.sidebar.button("Upgrade to Paid", on_click=upgrade)

# Ingredient input
st.sidebar.subheader("1️⃣ Ingredients Input")
uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])
manual = st.sidebar.text_input("Or type ingredients (comma-separated)")
# top_k if image
top_k = None
if uploaded:
    top_k = st.sidebar.slider("How many to detect?", 1, len(candidate_labels), 5)

# Preferences loaded for paid
spice_level = 5
cuisine = "Indian"
cook_time = "Easy (10-15 min)"
health_goals = ["General Health"]
if st.session_state.user and st.session_state.subscription == "Paid":
    # load existing
    c.execute(
        "SELECT spice_level,cuisine,cook_time,health_goals FROM preferences WHERE user_id=?",
        (st.session_state.user_id,),
    )
    pr = c.fetchone()
    if pr:
        spice_level, cuisine, cook_time, goals = pr
        health_goals = goals.split(",")
    st.sidebar.markdown("---")
    st.sidebar.subheader("2️⃣ Preferences")

# Define lists for selectboxes
cuisine_list = [
    "Indian",
    "Mexican",
    "Italian",
    "Chinese",
    "Thai",
    "American",
    "Mediterranean",
]
cook_list = ["Easy (10-15 min)", "Medium (15-20 min)", "Hard (20-25 min)"]

# Select existing or default
spice_level = st.sidebar.slider("Spice Level", 0, 10, spice_level)
cuisine = st.sidebar.selectbox(
    "Cuisine",
    cuisine_list,
    index=cuisine_list.index(cuisine) if cuisine in cuisine_list else 0,
)
cook_time = st.sidebar.selectbox(
    "Difficulty & Time",
    cook_list,
    index=cook_list.index(cook_time) if cook_time in cook_list else 0,
)
health_goals = st.sidebar.multiselect(
    "Health Goals",
    [
        "General Health",
        "Heart Health",
        "Diabetes-Friendly",
        "Fatty Liver Recovery",
        "Weight Loss",
    ],
    default=health_goals,
)

if st.sidebar.button("Save Preferences"):
    prefs = ",".join(health_goals)
    c.execute(
        "REPLACE INTO preferences(user_id,spice_level,cuisine,cook_time,health_goals) VALUES(?,?,?,?,?)",
        (st.session_state.user_id, spice_level, cuisine, cook_time, prefs),
    )
    conn.commit()
    st.sidebar.success("Preferences saved.")

# Generate
if st.sidebar.button("🚀 Generate Recipe Plan"):
    st.session_state.active = "Results"

# ─── Main ──────────────────────────────────────────────────────────
if st.session_state.active == "Input":
    st.header("🥗 Veggie Detector & Recipe Builder")
    cols = st.columns(2)
    with cols[0]:
        if uploaded:
            st.image(uploaded, caption="Uploaded", use_column_width=True)
        else:
            st.info("Upload an image via the sidebar.")
    with cols[1]:
        if manual:
            st.markdown(f"**Manual:** {manual}")
elif st.session_state.active == "Results":
    st.header("📝 Recipe Details")
    final = []
    if uploaded and top_k:
        img = Image.open(uploaded).convert("RGB")
        final += [n for n, _ in detect_vegetables(img, candidate_labels, top_k)]
    if manual:
        final += [i.strip().lower() for i in manual.split(",") if i.strip()]
    final = list(dict.fromkeys(final))
    if final:
        st.subheader("Ingredients:")
        for i in final:
            st.write(f"• {i.title()}")
    else:
        st.error("No ingredients found.")
    if st.session_state.subscription == "Paid" and final:
        st.subheader("Preferences & Generated Recipe")
        st.write(f"Spice: {spice_level}/10 | Cuisine: {cuisine} | Time: {cook_time}")
        st.write(f"Health Goals: {', '.join(health_goals)}")
        st.text_area("Recipe:", "Your recipe will appear here.", height=200)
    if st.button("🔄 Back"):
        st.session_state.active = "Input"
