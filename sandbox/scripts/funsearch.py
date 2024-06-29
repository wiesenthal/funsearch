import sqlite3

conn = sqlite3.connect('/data/example.db')

def fetch_function(id: str):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM functions WHERE id = ?', (id,))
    return cursor.fetchone()

print(fetch_function(1))
