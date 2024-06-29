# funsearch

# Local Code Execution Sandbox with SQLite Database

This guide helps you set up a simple, local code execution sandbox using Docker. The sandbox includes a SQLite database for simple data storage.

## Prerequisites

- Docker installed on your machine. [Get Docker](https://docs.docker.com/get-docker/)

## Setup

### Make sure you execute all 'docker' commands from the /sandbox directory.

1. Navigate to the sandbox directory:

`cd sandbox`

2. Create a .env file in the sandbox directory with the following content:

`OPENAI_API_KEY=your_openai_api_key`

3. Build the Docker image:

`docker build -t code-sandbox .`

## Usage

Run a container from the image:

`docker run -it --rm -v $(pwd)/data:/data code-sandbox`

This command will:

1. Initialize the SQLite database (if it doesn't exist)
2. Start a Python interactive shell

The database will be stored in the `data` directory, which persists between container runs.

## Interacting with the Database

In the Python shell, you can interact with the SQLite database:

```python
import sqlite3

conn = sqlite3.connect('/data/example.db')
cursor = conn.cursor()

# Query the users table
cursor.execute("SELECT * FROM users")
print(cursor.fetchall())

conn.close()
```

### Customizing Initialization

You can modify the init.sh script in the project directory to customize the startup process or add more initialization steps.

### Maintenance

Rebuild the Docker image when you change the Dockerfile, .env, requirements.txt, or init.sh:

`docker build -t code-sandbox .`

If you update .env after building, you don't need to rebuild. The .env file is mounted at runtime.

### Note for Contributors

- Do not commit your .env file to the repository if it contains sensitive information.
- Update requirements.txt if you add new Python packages.
