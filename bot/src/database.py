###
# database.py
# manages insertions into the postegreSQL database
###

# external modules
import os
import asyncio
import asyncpg
from datetime import datetime

async def test_db(attachments, content):
    print("IN TEST DB")
    print(attachments)
    print(content)
    
    # get env to connect to db
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("database environment variable not found.")
        return
    # init empty
    conn = None

    try:
        # connect to db
        conn = await asyncpg.connect(database_url)
        
        # create a simple test table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS wrong (
                id SERIAL PRIMARY KEY,
                image_url TEXT,
                codes TEXT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        list_of_codes = [code for code in content.split("\n") if code.strip()]
        print(list_of_codes)
        
        insert_query = """
            INSERT INTO wrong (image_url, codes)
            VALUES ($1, $2)
        """
        await conn.execute(
            insert_query,
            attachments[0] if attachments else None,
            list_of_codes
        )

        # Query it back
        #rows = await conn.fetch("SELECT * FROM wrong")
        #for row in rows:
        #    print(f"ID: {row['id']}, Image: {row['image_url']}, Codes: {row['codes']}")
        print("Database entry successful!\n")
    
    except asyncpg.PostgresError as pg_err:
        print(f"Database PostgreSQLError {pg_err}")
        print(f"Query arguments: $1='{attachments[0] if attachments else None}', $2={list_of_codes if 'list_of_codes' in locals() else 'not parsed yet'}")
    except Exception as e:
        print(f"Database entry failed: {e}\n")
    finally:
        if conn:
            await conn.close()


