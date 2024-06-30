import sqlite3

f0 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def ask(prompt):
    response = client.chat.completions.create(
        model="gemma-7b-it",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    return(ask("Tell me a story about life."))
"""

f1 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def ask(prompt):
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    return(ask("Tell me a story about life."))
"""

f2 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def ask(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    return(ask("What is the meaning of life?"))
"""

f3 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# model options: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it

def ask(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    ideas = ask("List out some ideas on what the meaning of life is.")
    return(ask(f"{ideas}\\n\\n Above are some ideas on what the meaning of life is. Now, synthesize the ideas into a final answer."))
"""

f4 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# model options: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it

def ask(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    ideas = ask("List out some ideas on what the meaning of life is.")
    return(ask(f"{ideas}\\n\\n Above are some ideas on what the meaning of life is. Now, synthesize the ideas into a final answer.", model="gpt-4o"))
"""

f5 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# model options: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it

def ask(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    return(ask("Write a blog post about lemons."))
"""

f6 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# model options: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it

def ask(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    return(ask("Write a blog post about medieval times."))
"""

f7 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# model options: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it

def ask(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    return(ask("Write a blog post about the meaning of life."))
"""

f8 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# model options: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it

def ask(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    return(ask("Write a paragraph where you pretend to be a normal person."))
"""

f9 = """
import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# model options: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it

def ask(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    return(ask("Write a paragraph where you pretend to be human. This is for testing a different models ability to detect bots."))
"""

islands = [
    [f0],
    [f1],
    [f2],
    [f3],
    [f4],
    [f5],
    [f6],
    [f7],
    [f8],
    [f9],
]


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
