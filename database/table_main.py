import psycopg2

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="regdb",
    user= "postgres",
    password= "postgres",
    host="172.18.0.3",
    port='5432'
)
cursor = conn.cursor()

# Create the table
# cursor.execute(
#     """
#         CREATE TABLE person_embeddings (
#             national_id VARCHAR(255) PRIMARY KEY,
#             person_name VARCHAR(255) NOT NULL,
#             embedding_1 VECTOR(512) NOT NULL,
#             embedding_2 VECTOR(512) NOT NULL,
#             embedding_3 VECTOR(512) NOT NULL,
#             embedding_4 VECTOR(512) NOT NULL,
#             embedding_5 VECTOR(512) NOT NULL
#         );

#     """
# )


# Create the table
cursor.execute(
    """
        CREATE TABLE detected_people (
            id SERIAL PRIMARY KEY,
            cam_id INTEGER NOT NULL,
            face_path VARCHAR(255) NOT NULL,
            frame_path VARCHAR(255) NOT NULL,
            face_embed VECTOR(512) NOT NULL,
            face_score FLOAT NOT NULL,
            datetime TIMESTAMP NOT NULL,
            matched BOOLEAN NOT NULL,
            national_id VARCHAR(255) DEFAULT NULL,
            person_name VARCHAR(255) DEFAULT NULL,
            similarity FLOAT DEFAULT NULL,
            FOREIGN KEY (national_id) REFERENCES person_embeddings(national_id)
        );

    """
)

# Commit the transaction
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()