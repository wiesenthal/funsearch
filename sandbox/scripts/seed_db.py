import sqlite3

f1 = """
from openai import OpenAI
client = OpenAI()

def ask(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    print(ask("What is the meaning of life?"))

main()
"""

f2 = """
from openai import OpenAI
client = OpenAI()

def ask(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    ideas = ask("List out some ideas on what the meaning of life is.")
    print(ask(f"{ideas}\n\n Above are some ideas on what the meaning of life is. Now, synthesize the ideas into a final answer."))
    

main()
"""


def seed_db():
    conn = sqlite3.connect("/data/functions.db")
    cursor = conn.cursor()

    # insert 2 functions, if they don't exist
    cursor.execute(
        """
    INSERT OR IGNORE INTO functions (island_id, timestamp_generated, status, code)
    VALUES (?, ?, ?, ?)
    """,
        (1, "2023-05-24 12:00:00", "active", f1),
    )
    cursor.execute(
        """
    INSERT OR IGNORE INTO functions (island_id, timestamp_generated, status, code)
    VALUES (?, ?, ?, ?)
    """,
        (2, "2023-05-24 12:00:00", "active", f2),
    )
    conn.commit()

    conn.close()

    print("Database initialized successfully.")


if __name__ == "__main__":
    seed_db()
