# === app.py ===
import streamlit as st
from db import get_db_connection
from auth import register_user, login_user, load_preferences, save_preferences
from detect import detect_vegetables, candidate_labels
from components import login_form, preferences_form, ingredient_input
from PIL import Image
from recipe_gen import generate_recipe  # accepts recipe_name parameter

# --- Streamlit Page Config ---
st.set_page_config(page_title="IngrEdibles", layout="wide")

# --- Sleek Professional Theme (Subtle Yellow & Red, Black Secondary) ---
st.markdown(
    """
    <style>
    /* App background */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }
    /* Sidebar background & text */
    [data-testid="stSidebar"] {
        background-color: #FFC72C;
        color: #ffffff;
    }
    /* Tab headers */
    .stTabs [role="tab"] {
        color: #FFC72C;
        font-weight: 600;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        border-bottom: 2px solid #DA291C;
    }
    /* Main headers */
    h1, h2, h3, h4, h5 {
        color: #FFC72C;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Buttons */
    .stButton>button {
        background-color: #DA291C !important;
        color: #ffffff !important;
        border-radius: 4px;
        font-weight: 600;
        border: none;
        padding: .375rem .75rem;
    }
    .stButton>button:hover {
        background-color: #660000 !important;
    }
    /* Card styling */
    .card {
        background-color: #FFEB3B;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #E53935;
        padding: 1rem;
    }
    /* Links and icons hover */
    a img:hover {
        filter: brightness(0.85);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize DB Connection ---
conn = get_db_connection()

# --- Top-right icons: Chart & Cart ---
top_bar_html = """
<div style="position:fixed; top:10px; right:20px; z-index:1000; display:flex; gap:12px;">
  <a href="/analytics" title="View Analytics">
    <img src="https://img.icons8.com/ios-filled/40/000000/combo-chart.png" alt="Chart"/>
  </a>
  <a href="/cart" title="View Cart">
    <img src="https://img.icons8.com/ios-filled/40/000000/shopping-cart.png" alt="Cart"/>
  </a>
</div>
"""
st.markdown(top_bar_html, unsafe_allow_html=True)

# --- Session Defaults ---
if "user" not in st.session_state:
    st.session_state.user = None
if "subscription" not in st.session_state:
    st.session_state.subscription = "Free"

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Home", "Preferences", "Profile"])

# --- Tab 1: Home ---
with tab1:
    st.header("IngrEdible AI")
    if not st.session_state.user:
        st.info("Please log in on the Profile tab to use detection features.")
    else:
        st.markdown("Upload a photo or enter the ingredients to get started.")
        col_img, col_input = st.columns([2, 1])
        with col_input:
            uploaded, manual, top_k = ingredient_input()
            # Optional recipe name for paid users
            if st.session_state.subscription == "Paid":
                recipe_name = st.text_input("Recipe Name (optional)")
            else:
                recipe_name = None
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            if st.button("Generate Plan", key="home_generate"):
                ingredients = []
                if uploaded and top_k:
                    img = Image.open(uploaded).convert("RGB")
                    ingredients += [
                        n for n, _ in detect_vegetables(img, candidate_labels, top_k)
                    ]
                if manual:
                    ingredients += [
                        i.strip().lower() for i in manual.split(",") if i.strip()
                    ]
                st.session_state.detected = list(dict.fromkeys(ingredients))
            if uploaded:
                st.image(uploaded, caption="Image Preview", width=200)

        with col_img:
            if "detected" in st.session_state:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                if st.session_state.detected:
                    st.subheader("Detected Ingredients:")
                    for ing in st.session_state.detected:
                        st.write(f"• {ing.title()}")
                else:
                    st.write("No ingredients detected. Please try again.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Detection results will appear here.")

        if "detected" in st.session_state:
            st.markdown("---")
            st.subheader("Recipe Suggestions")
            ingredients = st.session_state.detected
            ing_list = ", ".join([i.title() for i in ingredients])
            try:
                if st.session_state.subscription == "Paid":
                    prefs = load_preferences(st.session_state.user_id) or {}
                    info = (
                        f"Spice: {prefs.get('spice_level',5)}/10 | "
                        f"Serving: {prefs.get('serving',2)} | "
                        f"Cuisine: {prefs.get('cuisine','any')} | "
                        f"Meal Type: {prefs.get('meal_type','any')} | "
                        f"Time: {prefs.get('cook_time','any')}"
                    )
                    st.markdown(f"**Personalized Settings:** {info}")
                    pref_items = [f"{k}: {v}" for k, v in prefs.items() if v]
                    pref_str = "; ".join(pref_items)
                    recipe = generate_recipe(
                        ingredients_list=ing_list.lower(),
                        cuisine=prefs.get("cuisine", "any"),
                        difficulty=prefs.get("cook_time", "any"),
                        meal=prefs.get("meal_type", "any"),
                        preferences=pref_str,
                        recipe_name=recipe_name,
                    )
                    st.markdown(recipe, unsafe_allow_html=True)
                else:
                    recipe = generate_recipe(
                        ingredients_list=ing_list.lower(),
                        cuisine="any",
                        difficulty="any",
                        meal="any",
                        preferences="",
                        recipe_name=None,
                    )
                    st.text_area("General Recipe:", recipe, height=400)
            except Exception as e:
                st.error(f"Error generating recipe: {e}")

# --- Tab 2: Preferences ---
with tab2:
    st.header("Manage Your Preferences")
    if not st.session_state.user:
        st.info("Please log in on the Profile tab to view or edit preferences.")
    elif st.session_state.subscription == "Paid":
        existing = load_preferences(st.session_state.user_id) or {}
        if existing:
            with st.expander("Your Saved Preferences", expanded=True):
                st.write(f"- **Spice Level:** {existing.get('spice_level',5)}/5")
                st.write(f"- **Serving:** {existing.get('serving',2)}")
                st.write(f"- **Cuisine:** {existing.get('cuisine','Indian')}")
                st.write(f"- **Meal Type:** {existing.get('meal_type','Lunch')}")
                st.write(
                    f"- **Cook Time:** {existing.get('cook_time','Easy (10-15 min)')}"
                )
                st.write(
                    f"- **Health Goals:** {', '.join(existing.get('health_goals',[]))}"
                )
        defaults = {
            "cuisine_list": [
                "Indian",
                "Mexican",
                "Italian",
                "Chinese",
                "Thai",
                "American",
                "Mediterranean",
            ],
            "cook_list": ["Easy (10-15 min)", "Medium (15-20 min)", "Hard (20-25 min)"],
            "health_options": [
                "General Health",
                "Heart Health",
                "Diabetes-Friendly",
                "Fatty Liver Recovery",
                "Weight Loss",
            ],
            "meal_type": ["Breakfast", "Lunch", "Dinner", "Snack"],
        }
        new_prefs = preferences_form({**defaults, **existing}, disabled=False)
        if new_prefs:
            save_preferences(st.session_state.user_id, new_prefs)
            st.success("Preferences saved.")
    else:
        st.info("Upgrade to Paid to set your recipe preferences.")
        if st.button("Upgrade Now", key="pref_upgrade"):
            conn.execute(
                "UPDATE users SET subscription='Paid' WHERE id=?",
                (st.session_state.user_id,),
            )
            conn.commit()
            st.session_state.subscription = "Paid"
            st.success("Upgraded to Paid! You can now set preferences.")

# --- Tab 3: Profile ---
with tab3:
    st.header("👤 Your Profile")
    profile_col, action_col = st.columns([1, 2])
    with profile_col:
        st.markdown(
            "<div class='card' style='text-align:center;'>", unsafe_allow_html=True
        )
        avatar_url = "https://img.icons8.com/ios-filled/100/E53935/user-male-circle.png"
        st.image(avatar_url, width=40)
        username = st.session_state.user or "Guest"
        st.markdown(
            f"<h3 style='color:#000000'>{username}</h3>", unsafe_allow_html=True
        )
        sub = st.session_state.subscription
        st.markdown(
            f"<p style='color:#555555'>Subscription: <strong>{sub}</strong></p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with action_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if not st.session_state.user:
            st.subheader("Login / Register")
            mode, uname, pwd, submitted = login_form()
            if submitted:
                if mode == "Register":
                    ok = register_user(uname, pwd)
                    if ok:
                        st.success("Registered! Please log in.")
                    else:
                        st.error("Username taken.")
                else:
                    info = login_user(uname, pwd)
                    if info:
                        st.session_state.user = info["username"]
                        st.session_state.user_id = info["id"]
                        st.session_state.subscription = info["subscription"]
                        st.success(f"Welcome, {uname}!")
                    else:
                        st.error("Login failed.")
        else:
            st.subheader("Account Settings")
            if st.session_state.subscription == "Paid":
                if st.button("Cancel Subscription"):
                    conn.execute(
                        "UPDATE users SET subscription='Free' WHERE id=?",
                        (st.session_state.user_id,),
                    )
                    conn.commit()
                    st.session_state.subscription = "Free"
                    st.success("Subscription canceled.")
            else:
                if st.button("Upgrade to Paid"):
                    conn.execute(
                        "UPDATE users SET subscription='Paid' WHERE id=?",
                        (st.session_state.user_id,),
                    )
                    conn.commit()
                    st.session_state.subscription = "Paid"
                    st.success("Upgraded to Paid!")
            st.markdown("<hr style='border:1px solid #EEE'>", unsafe_allow_html=True)
            if st.button("Logout"):
                for key in ["user", "user_id", "subscription", "detected"]:
                    st.session_state.pop(key, None)
                st.success("You have been logged out.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Footer with social media icons ---
footer_html = """
<hr style="margin-top:2rem;"/>
<div style="text-align:center; padding:1rem 0;">
  <a href="https://twitter.com/YourProfile" target="_blank" style="margin:0 8px;">
    <img src="https://img.icons8.com/ios-glyphs/24/000000/twitter.png" alt="Twitter"/>
  </a>
  <a href="https://facebook.com/YourProfile" target="_blank" style="margin:0 8px;">
    <img src="https://img.icons8.com/ios-glyphs/24/000000/facebook-new.png" alt="Facebook"/>
  </a>
  <a href="https://instagram.com/YourProfile" target="_blank" style="margin:0 8px;">
    <img src="https://img.icons8.com/ios-glyphs/24/000000/instagram-new.png" alt="Instagram"/>
  </a>
  <a href="https://linkedin.com/in/YourProfile" target="_blank" style="margin:0 8px;">
    <img src="https://img.icons8.com/ios-glyphs/24/000000/linkedin-circled.png" alt="LinkedIn"/>
  </a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)


# === components.py ===
import streamlit as st
from PIL import Image
from typing import List, Tuple
from detect import candidate_labels

# --- Repeat Theme CSS for Sidebar Elements ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #FFC72C;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Login / Register form
def login_form():
    st.sidebar.header("Account")
    mode = st.sidebar.radio("Mode", ["Login", "Register"], index=0)
    username = st.sidebar.text_input("Username", key="auth_user")
    password = st.sidebar.text_input("Password", type="password", key="auth_pw")
    submit = st.sidebar.button(mode)
    return mode, username, password, submit


# Preferences form
def preferences_form(prefs: dict, disabled: bool):
    st.sidebar.subheader("Preferences")
    spice = st.sidebar.slider(
        "Spice Level", 1, 5, prefs.get("spice_level", 3), disabled=disabled
    )
    serving = st.sidebar.slider(
        "Serving", 1, 5, prefs.get("serving", 2), disabled=disabled
    )
    cuisine_list = prefs.get("cuisine_list", [])
    cuisine = st.sidebar.selectbox(
        "Cuisine Type",
        cuisine_list,
        index=(
            cuisine_list.index(prefs.get("cuisine", cuisine_list[0]))
            if prefs.get("cuisine") in cuisine_list
            else 0
        ),
        disabled=disabled,
    )
    cook_list = prefs.get("cook_list", [])
    cook = st.sidebar.selectbox(
        "Cook Time",
        cook_list,
        index=(
            cook_list.index(prefs.get("cook_time", cook_list[0]))
            if prefs.get("cook_time") in cook_list
            else 0
        ),
        disabled=disabled,
    )
    health_options = prefs.get("health_options", [])
    health = st.sidebar.multiselect(
        "Health Goals",
        health_options,
        default=prefs.get("health_goals", []),
        disabled=disabled,
    )
    meal_options = prefs.get("meal_options", ["Breakfast", "Lunch", "Dinner", "Snack"])
    meal_type = st.sidebar.selectbox(
        "Meal Type",
        meal_options,
        index=(
            meal_options.index(prefs.get("meal_type", meal_options[0]))
            if prefs.get("meal_type") in meal_options
            else 0
        ),
        disabled=disabled,
    )
    if st.sidebar.button("Save Preferences") and not disabled:
        return {
            "spice_level": spice,
            "meal_type": meal_type,
            "serving": serving,
            "cuisine": cuisine,
            "cook_time": cook,
            "health_goals": health,
        }
    return None


# Ingredient input component
def ingredient_input():
    st.sidebar.subheader("Ingredients")
    uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])
    manual = st.sidebar.text_input("Or Enter the ingredients", "")
    top_k = None
    if uploaded:
        top_k = st.sidebar.slider("How many to detect?", 1, len(candidate_labels), 5)
    return uploaded, manual, top_k
