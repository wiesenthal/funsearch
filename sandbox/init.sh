#!/bin/bash
set -e

# Initialize the database
python scripts/init_db.py

# Seed the database
python scripts/seed_db.py

# Load environment variables from .env file
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

bash
