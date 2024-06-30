import random
import sqlite3
import time
from typing import Optional
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout
import math

conn = sqlite3.connect("/data/programs.db")

from openai import OpenAI

client = OpenAI()


def ask(prompt, model="gpt-4o", temperature=0.0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    return response.choices[0].message.content


class Program:
    def __init__(
        self,
        id: int,
        island_id: int,
        created_at: str,
        status: str,
        code: str,
        score: float,
        execution_time: float,
    ):
        self.id = id
        self.island_id = island_id
        self.created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        self.status = status
        self.code = code
        self.score = score
        self.execution_time = execution_time

    def save(self):
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE programs SET score = ?, execution_time = ? WHERE id = ?",
            (
                self.score,
                self.execution_time,
                self.id,
            ),
        )
        conn.commit()

    def __str__(self):
        return f"Program {self.id} with score {self.score} and execution time {self.execution_time}. Code:\n{self.code}"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, island_id={self.island_id}, created_at={self.created_at}, status={self.status}, score={self.score}, execution_time={self.execution_time})"


def get_island_ids() -> list[int]:
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT island_id FROM programs")
    return [row[0] for row in cursor.fetchall()]


def sample_island_id() -> int:
    return random.choice(get_island_ids())


def calculate_cluster_probability(scores, n, N, T0):
    def calculate_T_cluster(n, N, T0):
        return T0 * (1 - ((n % N) / N))

    T_cluster = calculate_T_cluster(n, N, T0)

    exp_scores = [math.exp(s / T_cluster) for s in scores]
    total_exp_score = sum(exp_scores)

    probabilities = [exp_score / total_exp_score for exp_score in exp_scores]

    return probabilities


# Example usage:
# scores = [s1, s2, s3, ...]
# n = current iteration
# N = total number of iterations
# T0 = initial temperature
# probabilities = calculate_cluster_probability(scores, n, N, T0)


def fetch_program_clusters(island_id: int) -> dict[float, list[Program]]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM programs WHERE island_id = ? ORDER BY score DESC",
        (island_id,),
    )
    results = cursor.fetchall()

    clusters = {}
    if results:
        for result in results:
            program = Program(*result)
            score = program.score
            if score not in clusters:
                clusters[score] = []
            clusters[score].append(program)
    return clusters


MIN_CHARS = 100


def evaluate_output(output: str, retries: int = 3) -> float:
    evaluation_prompt = """\
Your performance at detecting bots is critical to humanitarian success.\
For the sake of parsing, only output a float between 0 and 1, with no other text.\
From 0 to 1, output a decimal representing the probability that the following output was written by a bot:\n"""

    if len(output) < MIN_CHARS:
        return 0.0

    while retries > 0:
        try:
            response = ask(evaluation_prompt + f"{output}")
            num = float(response)
            if num < 0 or num > 1:
                raise ValueError(f"Expected a number between 0 and 1, got {num}")
            return round(1 - num, 2)
        except Exception as e:
            print(f"Error evaluating output: {e}")
            retries -= 1
    return 0.5


def execute_code(program: Program) -> str:
    with redirect_stdout(StringIO()) as stdout:
        namespace = {}
        exec(program.code, namespace)
        return stdout.getvalue()


island_id = sample_island_id()
print(f"Island ID: {island_id}")
clusters = fetch_program_clusters(island_id)
print(f"Clusters: {clusters}")


# Example usage
# program = fetch_program(2)
# if program:
#     print(f"Code:\n{program.code}")
#     print(f"Executing program...")
#     t1 = time.time()
#     result = execute_code(program)
#     t2 = time.time()
#     print(f"Result: {result}.\nTime taken: {t2 - t1} seconds")
#     print(f"Evaluating result...")
#     t3 = time.time()
#     evaluation = evaluate_output(result)
#     t4 = time.time()
#     print(f"Evaluation: {evaluation}.\nTime taken: {t4 - t3} seconds")
#     program.score = evaluation
#     program.execution_time = t2 - t1
#     program.save()
# else:
#     print("Program not found")
