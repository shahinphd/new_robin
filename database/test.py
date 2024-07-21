import psycopg2
import numpy as np

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="regdb",
    user="postgres",
    password="postgres",
    host="172.18.0.2",
    port=5432
)
cursor = conn.cursor()

# Example vector to compare
compare_vector = np.random.rand(512).tolist()
# print(compare_vector)

# Create a string representation of the vector
compare_vector_str = compare_vector
compare_vector_str = '[' + ', '.join(map(str, compare_vector)) + ']'

# Find the most similar embedding for each person
query = """
    WITH similarities AS (
        SELECT
            person_id,
            LEAST(
                %s <-> embedding_1,
                %s <-> embedding_2,
                %s <-> embedding_3,
                %s <-> embedding_4,
                %s <-> embedding_5
            ) AS similarity
        FROM person_embeddings
    )
    SELECT person_id, similarity
    FROM similarities
    ORDER BY similarity
    LIMIT 1;
"""

# Execute the query with the vector as a parameter
cursor.execute(query, (compare_vector_str, compare_vector_str, compare_vector_str, compare_vector_str, compare_vector_str))
result = cursor.fetchone()

# Output the most similar embedding
print("Most similar person_id:", result[0])
print("Similarity:", result[1])

# Close the connection
cursor.close()
conn.close()
