import streamlit as st
from db import get_db_connection
from auth import register_user, login_user, load_preferences, save_preferences
from detect import detect_vegetables, candidate_labels
from components import login_form, preferences_form, ingredient_input
from PIL import Image

# --- Initialize DB Connection ---
conn = get_db_connection()

# --- Page Configuration ---
st.set_page_config(page_title="VeggieChef", page_icon="🥗", layout="wide")

# Apply modern theme with CSS
st.markdown(
    """
    <style>
      .css-18e3th9 { font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', sans-serif; }
      .block-container { padding: 2rem 4rem; background-color: #F9F9F9; }
      .stTabs [role="tab"] { padding: 0.75rem 1.5rem; font-size: 1.1rem; }
      .stTabs [role="tabSelected"] { color: #DA291C; border-bottom: 3px solid #DA291C; }
      .card { background: #FFF; border-radius: 1rem; padding: 1.5rem; box-shadow: 0 2px 12px rgba(0,0,0,0.1); margin-bottom: 2rem; }
      .stButton>button { background: #FFC72C; color: #DA291C; border-radius: 0.75rem; padding: 0.75rem 1.5rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session Defaults ---
if "user" not in st.session_state:
    st.session_state.user = None
if "subscription" not in st.session_state:
    st.session_state.subscription = "Free"

# --- Navigation Tabs ---
tab1, tab2, tab3 = st.tabs(["🏠 Home", "⚙️ Preferences", "👤 Profile"])

# --- Tab 1: Home ---
with tab1:
    st.header("Veggie Detection & Recipe Plan")
    if not st.session_state.user:
        st.info("Please log in on the Profile tab to use detection features.")
    else:
        st.markdown("Upload a photo or type ingredients to get started.")
        col_img, col_input = st.columns([2, 1])
        with col_input:
            # Input controls
            uploaded, manual, top_k = ingredient_input()

            # Spacer to keep button at consistent position
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            # Generate Plan button always at bottom of this panel
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
            # Preview uploaded image if any
            if uploaded:
                st.image(
                    uploaded, caption="Image Preview", use_column_width=False, width=40
                )
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

        # Recipe suggestions for all users
        if "detected" in st.session_state:
            st.markdown("---")
            st.subheader("Recipe Suggestions")
            ingredients = st.session_state.detected
            ing_list = ", ".join([i.title() for i in ingredients])
            if st.session_state.subscription == "Paid":
                prefs = load_preferences(st.session_state.user_id) or {}
                info = (
                    f"Spice: {prefs.get('spice_level',5)}/10 | "
                    f"Cuisine: {prefs.get('cuisine','Indian')} | "
                    f"Time: {prefs.get('cook_time','Easy (10-15 min)')}"
                )
                st.markdown(f"**Personalized Settings:** {info}")
                st.text_area(
                    "Your Recipe:",
                    f"Here is a personalized recipe for {ing_list} based on your preferences.",
                    height=200,
                )
            else:
                st.text_area(
                    "General Recipe:",
                    f"Here is a general recipe you can try with {ing_list}.",
                    height=200,
                )

# --- Tab 2: Preferences ---
with tab2:
    st.header("Manage Your Preferences")
    if not st.session_state.user:
        st.info("Please log in on the Profile tab to view or edit preferences.")
    elif st.session_state.subscription == "Paid":
        existing = load_preferences(st.session_state.user_id) or {}
        if existing:
            with st.expander("Your Saved Preferences", expanded=True):
                st.write(f"- **Spice Level:** {existing.get('spice_level',5)}/10")
                st.write(f"- **Cuisine:** {existing.get('cuisine','Indian')}  ")
                st.write(
                    f"- **Cook Time:** {existing.get('cook_time','Easy (10-15 min)')}  "
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
        avatar_url = "https://img.icons8.com/ios-filled/100/DA291C/user-male-circle.png"
        st.image(avatar_url, width=40)
        username = st.session_state.user or "Guest"
        st.markdown(f"<h3 style='color:#333'>{username}</h3>", unsafe_allow_html=True)
        sub = st.session_state.subscription
        st.markdown(
            f"<p style='color:#666'>Subscription: <strong>{sub}</strong></p>",
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
                    (
                        st.success("Registered! Please log in.")
                        if ok
                        else st.error("Username taken.")
                    )
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
