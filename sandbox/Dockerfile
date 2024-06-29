FROM python:3.9-slim

WORKDIR /sandbox

# Install SQLite
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

# Copy necessary files
COPY requirements.txt .env ./
COPY scripts scripts
COPY init.sh ./

# Install pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables from .env
ENV $(cat .env | xargs)
# Create a directory for the database
RUN mkdir /data

# Make sure the script is executable
RUN chmod +x init.sh

# Set the entrypoint to our init script
ENTRYPOINT ["/sandbox/init.sh"]
