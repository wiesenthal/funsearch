import random
import sqlite3
import time
from typing import Optional
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout
import math
from contextlib import contextmanager
from openai import OpenAI
import threading
from queue import Queue


DB_PATH = "/data/programs.db"

client = OpenAI()


@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def ask(prompt, model="gpt-4o", temperature=0.0) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    return response.choices[0].message.content


def messages_ask(messages, model="gpt-4o", temperature=1.5) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def calculate_cluster_probability(scores, n, N, T0):
    def calculate_T_cluster(n, N, T0):
        return T0 * (1 - ((n % N) / N))

    T_cluster = calculate_T_cluster(n, N, T0)

    exp_scores = [math.exp(s / T_cluster) for s in scores]
    total_exp_score = sum(exp_scores)

    probabilities = [exp_score / total_exp_score for exp_score in exp_scores]

    return probabilities


class Program:
    @staticmethod
    def create(
        island_id: int,
        code: str,
        score,
        execution_time,
        parent_ids: set[int],
    ) -> "Program":
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO programs (island_id, code, score, execution_time) VALUES (?, ?, ?, ?)",
                (island_id, code, score, execution_time),
            )
            program_id = cursor.lastrowid
            conn.commit()

            cursor.execute("SELECT * FROM programs WHERE id = ?", (program_id,))
            row = cursor.fetchone()

            # Insert parent_ids into the parent_ids table
            for parent_id in parent_ids:
                cursor.execute(
                    "INSERT INTO parent_ids (program_id, parent_id) VALUES (?, ?)",
                    (program_id, parent_id),
                )
            conn.commit()

        return Program(
            id=row[0],
            island_id=row[1],
            created_at=row[2],
            code=row[3],
            score=row[4],
            execution_time=row[5],
        )

    def __init__(
        self,
        id: int,
        island_id: int,
        created_at: str,
        code: str,
        score: float,
        execution_time: float,
    ):
        self.id = id
        self.island_id = island_id
        self.created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        self.code = code
        self.score = score
        self.execution_time = execution_time

    def save(self):
        with get_db_connection() as conn:
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
        return f"{self.__class__.__name__}(id={self.id}, island_id={self.island_id}, created_at={self.created_at}, score={self.score}, execution_time={self.execution_time})"


def get_island_ids() -> list[int]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT island_id FROM programs WHERE dead = FALSE")
        return [row[0] for row in cursor.fetchall()]


def sample_island_id() -> int:
    return random.choice(get_island_ids())


def kill_worst_half_islands_and_reseed() -> list[int]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT island_id, AVG(score) as avg_score FROM programs WHERE dead = FALSE GROUP BY island_id ORDER BY avg_score ASC"
        )
        results = cursor.fetchall()
        half = len(results) // 2
        ids = [row[0] for row in results[:half]]
        for id in ids:
            cursor.execute(
                "UPDATE programs SET dead = TRUE, dead_time = ? WHERE island_id = ?",
                (datetime.now(), id),
            )

        conn.commit()
    return ids


def get_top_program_from_island(island_id: int) -> Program:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM programs WHERE island_id = ? AND dead = FALSE ORDER BY score DESC LIMIT 1",
            (island_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"No programs found for island {island_id}")
        return Program(
            id=row[0],
            island_id=row[1],
            created_at=row[2],
            code=row[3],
            score=row[4],
            execution_time=row[5],
        )


def kill_and_reseed():
    dead_islands = kill_worst_half_islands_and_reseed()
    remaining_islands = [id for id in get_island_ids() if id not in dead_islands]
    islands_to_sample_from = random.choices(remaining_islands, k=len(dead_islands))
    for i, (island_id, dead_island_id) in enumerate(
        zip(islands_to_sample_from, dead_islands)
    ):
        program = get_top_program_from_island(island_id)
        new_program = Program.create(
            island_id=dead_island_id,
            code=program.code,
            score=program.score,
            execution_time=program.execution_time,
            parent_ids={program.id},
        )
        new_program.save()


N = 30_000  # Assuming total number of iterations, adjust as needed
T_cluster = 0.1


def sample_cluster(clusters: dict[float, list[Program]]) -> list[Program]:
    global N, T_cluster
    # Calculate probabilities for each cluster
    scores = list(clusters.keys())
    n = sum([len(cluster) for cluster in clusters.values()])
    probabilities = calculate_cluster_probability(scores, n, N, T_cluster)

    # Normalize probabilities
    total_prob = sum(probabilities)
    normalized_probs = [p / total_prob for p in probabilities]

    # Select a cluster based on the calculated probabilities
    selected_score = random.choices(scores, weights=normalized_probs, k=1)[0]

    return clusters[selected_score]


T_program = 1.0


def sample_program(cluster: list[Program]) -> Program:
    global T_program

    # Calculate normalized lengths
    lengths = [len(program.code) for program in cluster]
    min_length = min(lengths)
    max_length = max(lengths)

    normalized_lengths = [
        (length - min_length) / (max_length + 1e-6) for length in lengths
    ]

    # Calculate weights based on normalized execution lengths
    weights = [
        math.exp(-normalized_length / T_program)
        for normalized_length in normalized_lengths
    ]

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]

    # Select a program based on the calculated weights
    return random.choices(cluster, weights=normalized_weights, k=1)[0]


def fetch_program_clusters(island_id: int) -> dict[float, list[Program]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM programs WHERE island_id = ? AND dead = FALSE ORDER BY score DESC",
            (island_id,),
        )
        results = cursor.fetchall()

    clusters = {}
    if results:
        for result in results:
            program = Program(
                id=result[0],
                island_id=result[1],
                created_at=result[2],
                code=result[3],
                score=result[4],
                execution_time=result[5],
            )
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
You are an expert at detecting LLM output, and you are extremely discerning.\
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


def reproduce(programs: list[Program]):
    examples = []

    sorted_programs = sorted(programs, key=lambda x: x.score)

    for i, program in enumerate(sorted_programs):
        examples.append({"role": "user", "content": f"v{i}"})
        examples.append({"role": "assistant", "content": program.code})

    new_program = messages_ask(
        messages=[
            {
                "role": "system",
                "content": "Output an improved version of the python program. Externally, the program will be exec'd, and the main function result used, so, your entire output must be valid python code. Feel free to provide comments to explain your thought process.",
            },
            *examples,
            {"role": "user", "content": f"v{len(programs)}"},
        ]
    )
    if "```python" in new_program:
        new_program = new_program.split("```python")[1].split("```")[0]
    elif "```" in new_program:
        new_program = new_program.split("```")[1].split("```")[0]
    return new_program


def execute_code(code: str) -> str:
    namespace = {}
    exec(code, namespace)
    if "main" in namespace and callable(namespace["main"]):
        return namespace["main"]()
    raise Exception("No main function found")


K = 2


def sample_execute_store(debug=False):
    island_id = sample_island_id()
    clusters = fetch_program_clusters(island_id)
    cluster = sample_cluster(clusters)
    parents = [sample_program(cluster) for _ in range(K)]
    if debug:
        print(f"Sampled parents: {parents}")
    t0 = time.time()
    new_code = reproduce(parents)
    t1 = time.time()
    if debug:
        print(f"Generation time: {round(t1 - t0, 2)} seconds")
    try:
        output = execute_code(new_code)
    except Exception as e:
        if debug:
            print(f"Error executing code: {e}")
        return None
    t2 = time.time()
    if debug:
        print(f"Execution time: {round(t2 - t1, 2)} seconds")
    evaluation = evaluate_output(output)
    if debug:
        print(f"Score: {evaluation}")
    p = Program.create(
        island_id,
        new_code,
        evaluation,
        t2 - t1,
        set([parent.id for parent in parents]),
    )
    p.save()
    if debug:
        print(f"New program saved: {repr(p)}")
    return p


MAX_CONCURRENCY = 20
INTERVAL = 5  # seconds


def worker(task_queue):
    while True:
        task_queue.get()
        try:
            result = sample_execute_store(debug=False)
            print(f"Success. Score: {result.score}" if result else "Error")
        except Exception as e:
            print(f"Error in worker: {e}")
        finally:
            task_queue.task_done()


def main():
    task_queue = Queue()

    # Start worker threads
    for _ in range(MAX_CONCURRENCY):
        thread = threading.Thread(target=worker, args=(task_queue,), daemon=True)
        thread.start()

    i = 0
    while True:
        if (i + 1) % 200 == 0:
            print(f"KILLING AND RESEEDING. Iteration: {i}")
            kill_and_reseed()
        elif len(task_queue.queue) < MAX_CONCURRENCY:
            print(f"ITERATION {i}")
            task_queue.put(i)  # Add a task to the queue
            i += 1
            time.sleep(INTERVAL)  # Wait for 1 second before adding the next task
        else:
            print(f"Queue full. Waiting for {INTERVAL} seconds.")
            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
