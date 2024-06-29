import sqlite3


def init_db():
    conn = sqlite3.connect("/data/functions.db")
    cursor = conn.cursor()

    # Create the functions table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS functions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        island_id INTEGER NOT NULL,
        timestamp_generated DATETIME NOT NULL,
        status TEXT NOT NULL,
        code TEXT NOT NULL
    )
    """
    )

    # Create the executions table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        function_id INTEGER NOT NULL,
        start_time DATETIME NOT NULL,
        end_time DATETIME NOT NULL,
        score REAL,
        code TEXT NOT NULL,
        FOREIGN KEY (function_id) REFERENCES functions(id)
    )
    """
    )

    # Create the parent_ids table for many-to-many relationship
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS parent_ids (
        function_id INTEGER,
        parent_id INTEGER,
        FOREIGN KEY (function_id) REFERENCES functions(id),
        FOREIGN KEY (parent_id) REFERENCES functions(id),
        PRIMARY KEY (function_id, parent_id)
    )
    """
    )

    conn.commit()
    conn.close()

    print("Database initialized successfully.")


if __name__ == "__main__":
    init_db()
