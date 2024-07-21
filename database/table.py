import psycopg2

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="regdb",
    user= "postgres",
    password= "postgres",
    host="172.18.0.2",
    port='5432'
)
cursor = conn.cursor()

# Create the table
cursor.execute(
    """
        CREATE TABLE person_embeddings (
            id SERIAL PRIMARY KEY,
            person_name VARCHAR(255) NOT NULL,
            embedding_1 VECTOR(512) NOT NULL,
            embedding_2 VECTOR(512) NOT NULL,
            embedding_3 VECTOR(512) NOT NULL,
            embedding_4 VECTOR(512) NOT NULL,
            embedding_5 VECTOR(512) NOT NULL
        );

    """
)

# Commit the transaction
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()