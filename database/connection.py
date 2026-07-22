import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_connection():
    """
    Create and return a PostgreSQL database connection.
    """

    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

        print("✅ Connected to PostgreSQL successfully!")

        return conn

    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        return None


if __name__ == "__main__":
    conn = get_connection()

    if conn:
        conn.close()
        print("✅ Connection Closed.")