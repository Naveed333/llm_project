import sqlite3


# Initialize SQLite database and tables
def get_db_connection(db_path: str = "users.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY,
      username TEXT UNIQUE,
      password_hash TEXT,
      subscription TEXT DEFAULT 'Free'
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
    return conn
