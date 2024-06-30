import sqlite3

f0 = """
from openai import OpenAI
client = OpenAI()

def ask(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    print(ask("Tell me a story about life."))

main()
"""

f1 = """
from openai import OpenAI
client = OpenAI()

def ask(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
"""

f2 = """
from openai import OpenAI
client = OpenAI()

def ask(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    print(ask("What is the meaning of life?", model="gpt-3.5-turbo"))

main()
"""

f3 = """
from openai import OpenAI
client = OpenAI()

def ask(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    ideas = ask("List out some ideas on what the meaning of life is.")
    print(ask(f"{ideas}\\n\\n Above are some ideas on what the meaning of life is. Now, synthesize the ideas into a final answer."))

main()
"""

f4 = """
from openai import OpenAI
client = OpenAI()

def ask(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    ideas = ask("List out some ideas on what the meaning of life is.", model="gpt-3.5-turbo")
    print(ask(f"{ideas}\\n\\n Above are some ideas on what the meaning of life is. Now, synthesize the ideas into a final answer.", model="gpt-4o"))

main()
"""

island_1 = [f0, f1, f2]

island_2 = [f3, f4]

islands = [island_1, island_2]


def seed_db():
    conn = sqlite3.connect("/data/programs.db")
    cursor = conn.cursor()

    program_id = 0
    for island_id, programs in enumerate(islands):
        for program in programs:
            cursor.execute(
                "INSERT OR REPLACE INTO programs (id, island_id, code, score, execution_time) VALUES (?, ?, ?, ?, ?)",
                (program_id, island_id, program, 0.0, 1),
            )
            program_id += 1

    conn.commit()

    conn.close()

    print("Database initialized successfully.")


if __name__ == "__main__":
    seed_db()
