import streamlit as st
from PIL import Image
from typing import List, Tuple
from detect import candidate_labels


# Login / Register form
def login_form():
    st.sidebar.header("🔒 Account")
    mode = st.sidebar.radio("Mode", ["Login", "Register"], index=0)
    username = st.sidebar.text_input("Username", key="auth_user")
    password = st.sidebar.text_input("Password", type="password", key="auth_pw")
    submit = st.sidebar.button(mode)
    return mode, username, password, submit


# Preferences form
def preferences_form(prefs: dict, disabled: bool):
    st.sidebar.subheader("📋 Preferences")
    spice = st.sidebar.slider(
        "Spice Level", 1, 3, prefs.get("spice_level", 5), disabled=disabled
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
    manual = st.sidebar.text_input("Or type ingredients", "")
    top_k = None
    if uploaded:
        top_k = st.sidebar.slider("How many to detect?", 1, len(candidate_labels), 5)
    return uploaded, manual, top_k
