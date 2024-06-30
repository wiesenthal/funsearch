import sqlite3


def init_db():
    conn = sqlite3.connect("/data/programs.db")
    cursor = conn.cursor()

    # Create the programs table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS programs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        island_id INTEGER NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        code TEXT NOT NULL,
        score REAL NOT NULL,
        execution_time REAL NOT NULL,
        dead BOOLEAN NOT NULL DEFAULT FALSE,
        died_at DATETIME
    )
    """
    )

    # Create the parent_ids table for many-to-many relationship
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS parent_ids (
        program_id INTEGER,
        parent_id INTEGER,
        FOREIGN KEY (program_id) REFERENCES programs(id),
        FOREIGN KEY (parent_id) REFERENCES programs(id),
        PRIMARY KEY (program_id, parent_id)
    )
    """
    )

    conn.commit()
    conn.close()

    print("Database initialized successfully.")


if __name__ == "__main__":
    init_db()
