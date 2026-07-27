import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analyzer.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # 1. Uploads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            file_type TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            raw_text TEXT,
            processed_text TEXT
        )
    ''')
    
    # 2. Summaries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            summary_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (upload_id) REFERENCES uploads (id) ON DELETE CASCADE
        )
    ''')
    
    # 3. MCQs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mcqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            options TEXT NOT NULL, -- JSON string of list of options
            correct_answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (upload_id) REFERENCES uploads (id) ON DELETE CASCADE
        )
    ''')
    
    # 4. QA History table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (upload_id) REFERENCES uploads (id) ON DELETE CASCADE
        )
    ''')
    
    # 5. Keywords table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (upload_id) REFERENCES uploads (id) ON DELETE CASCADE
        )
    ''')
    
    # 6. Analytics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            difficulty_level TEXT NOT NULL,
            sentence_count INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            estimated_study_time INTEGER NOT NULL, -- in minutes
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (upload_id) REFERENCES uploads (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def save_upload(filename, filepath, file_type, raw_text=None, processed_text=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO uploads (filename, filepath, file_type, raw_text, processed_text) VALUES (?, ?, ?, ?, ?)",
        (filename, filepath, file_type, raw_text, processed_text)
    )
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return upload_id

def update_upload_texts(upload_id, raw_text, processed_text):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE uploads SET raw_text = ?, processed_text = ? WHERE id = ?",
        (raw_text, processed_text, upload_id)
    )
    conn.commit()
    conn.close()

def get_upload(upload_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_uploads():
    conn = get_db_connection()
    rows = conn.execute("SELECT id, filename, filepath, file_type, uploaded_at FROM uploads ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_upload(upload_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Delete physical file first
    row = cursor.execute("SELECT filepath FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if row and os.path.exists(row['filepath']):
        try:
            os.remove(row['filepath'])
        except OSError:
            pass
    cursor.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    conn.close()

def save_summary(upload_id, summary_text):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Delete existing summaries for this upload to avoid duplicate records
    cursor.execute("DELETE FROM summaries WHERE upload_id = ?", (upload_id,))
    cursor.execute(
        "INSERT INTO summaries (upload_id, summary_text) VALUES (?, ?)",
        (upload_id, summary_text)
    )
    conn.commit()
    conn.close()

def get_summary(upload_id):
    conn = get_db_connection()
    row = conn.execute("SELECT summary_text FROM summaries WHERE upload_id = ?", (upload_id,)).fetchone()
    conn.close()
    return row['summary_text'] if row else None

def save_mcqs(upload_id, mcqs_list):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM mcqs WHERE upload_id = ?", (upload_id,))
    for mcq in mcqs_list:
        cursor.execute(
            "INSERT INTO mcqs (upload_id, question, options, correct_answer) VALUES (?, ?, ?, ?)",
            (upload_id, mcq['question'], json.dumps(mcq['options']), mcq['correct_answer'])
        )
    conn.commit()
    conn.close()

def get_mcqs(upload_id):
    conn = get_db_connection()
    rows = conn.execute("SELECT question, options, correct_answer FROM mcqs WHERE upload_id = ?", (upload_id,)).fetchall()
    conn.close()
    mcqs = []
    for row in rows:
        mcqs.append({
            'question': row['question'],
            'options': json.loads(row['options']),
            'correct_answer': row['correct_answer']
        })
    return mcqs

def save_qa(upload_id, question, answer):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO qa_history (upload_id, question, answer) VALUES (?, ?, ?)",
        (upload_id, question, answer)
    )
    conn.commit()
    conn.close()

def get_qa_history(upload_id):
    conn = get_db_connection()
    rows = conn.execute("SELECT question, answer, created_at FROM qa_history WHERE upload_id = ? ORDER BY created_at ASC", (upload_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_keywords(upload_id, keywords_list):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM keywords WHERE upload_id = ?", (upload_id,))
    for kw, score in keywords_list:
        cursor.execute(
            "INSERT INTO keywords (upload_id, keyword, score) VALUES (?, ?, ?)",
            (upload_id, kw, score)
        )
    conn.commit()
    conn.close()

def get_keywords(upload_id):
    conn = get_db_connection()
    rows = conn.execute("SELECT keyword, score FROM keywords WHERE upload_id = ? ORDER BY score ASC", (upload_id,)).fetchall()
    conn.close()
    return [(row['keyword'], row['score']) for row in rows]

def save_analytics(upload_id, difficulty_level, sentence_count, word_count, estimated_study_time):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analytics WHERE upload_id = ?", (upload_id,))
    cursor.execute(
        "INSERT INTO analytics (upload_id, difficulty_level, sentence_count, word_count, estimated_study_time) VALUES (?, ?, ?, ?, ?)",
        (upload_id, difficulty_level, sentence_count, word_count, estimated_study_time)
    )
    conn.commit()
    conn.close()

def get_analytics(upload_id):
    conn = get_db_connection()
    row = conn.execute("SELECT difficulty_level, sentence_count, word_count, estimated_study_time FROM analytics WHERE upload_id = ?", (upload_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
