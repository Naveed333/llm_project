import sqlite3
import hashlib
from db import get_db_connection
import streamlit as st


# Password hashing
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# Registration
def register_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users(username,password_hash) VALUES(?,?)",
            (username, hash_password(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


# Login
def login_user(username: str, password: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id,password_hash,subscription FROM users WHERE username=?", (username,)
    )
    row = c.fetchone()
    if row and row[1] == hash_password(password):
        return {"id": row[0], "username": username, "subscription": row[2]}
    return None


# Preferences load/save
def load_preferences(user_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT spice_level,cuisine,cook_time,health_goals FROM preferences WHERE user_id=?",
        (user_id,),
    )
    row = c.fetchone()
    if row:
        return {
            "spice_level": row[0],
            "cuisine": row[1],
            "cook_time": row[2],
            "health_goals": row[3].split(","),
        }
    return None


def save_preferences(user_id: int, prefs: dict):
    conn = get_db_connection()
    c = conn.cursor()
    goals = ",".join(prefs["health_goals"])
    c.execute(
        "REPLACE INTO preferences(user_id,spice_level,cuisine,cook_time,health_goals) VALUES(?,?,?,?,?)",
        (user_id, prefs["spice_level"], prefs["cuisine"], prefs["cook_time"], goals),
    )
    conn.commit()
