
import numpy as np
import json
import pika
import os
import pandas as pd
import psycopg2

def send_to_queue(df, queue_name):
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    message = df.to_json()
    channel.basic_publish(exchange='', routing_key=queue_name, body=message)

    connection.close()

def compute_matching(channel, method, properties, body):
    conn = psycopg2.connect(
        dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT']
    )

    body = json.loads(body)
    df = pd.DataFrame.from_dict(body)
    df['Matched Name'] = None
    df['Matched ID'] = None
    df['Similarity'] = None
    df['Matched'] = False



    SIMILARITY_THRESHOLD = 1
    
    for idx, row in df.iterrows():
        cursor = conn.cursor()
        compare_vector_str = '[' + ', '.join(map(str, row['Feature'])) + ']'
        
        # Query to compare with recent entries
        query_recent = """
            WITH recent_entries AS (
                SELECT *
                FROM detected_people
                WHERE datetime >= NOW() - INTERVAL '10 minutes'
            ),
            similarities AS (
                SELECT
                    id,
                    cam_id,
                    face_path,
                    frame_path,
                    face_embed,
                    face_score,
                    datetime,
                    matched,
                    national_id,
                    person_name,
                    face_embed <-> %s AS similarity
                FROM recent_entries
            )
            SELECT *
            FROM similarities
            ORDER BY similarity
            LIMIT 1;
        """
        
        cursor.execute(query_recent, (compare_vector_str,))
        recent_match = cursor.fetchone()
        
        
        
        if recent_match and recent_match[10] <= SIMILARITY_THRESHOLD:  # assuming similarity is the 12th column
            # print( '*****recent_match[11]*******', recent_match[10], flush=True)
            df.at[idx, 'Matched Name'] = recent_match[9]
            df.at[idx, 'Matched ID'] = recent_match[8]
            df.at[idx, 'Similarity'] = float(recent_match[10])
            df.at[idx, 'Matched'] = True
            continue



        else:
            # Query to compare with the rest of the table
            query_rest = """
                WITH similarities AS (
                    SELECT
                        id,
                        cam_id,
                        face_path,
                        frame_path,
                        face_embed,
                        face_score,
                        datetime,
                        matched,
                        national_id,
                        person_name,
                        face_embed <-> %s AS similarity
                    FROM detected_people
                    WHERE datetime < NOW() - INTERVAL '10 minutes'
                )
                SELECT *
                FROM similarities
                ORDER BY similarity
                LIMIT 1;
            """
            
            cursor.execute(query_rest, (compare_vector_str,))
            rest_match = cursor.fetchone()

            
            if rest_match and rest_match[10] <= SIMILARITY_THRESHOLD:
                print( '*****rest_match*******', rest_match[10], flush=True)
                df.at[idx, 'Matched Name'] = rest_match[9]
                df.at[idx, 'Matched ID'] = rest_match[8]
                df.at[idx, 'Similarity'] = float(rest_match[10])
                df.at[idx, 'Matched'] = True
                continue

            else:
                print("No match found for vector", flush=True)
                query = """
                    WITH similarities AS (
                        SELECT
                            national_id,
                            person_name,
                            LEAST(
                                %s <-> embedding_1,
                                %s <-> embedding_2,
                                %s <-> embedding_3,
                                %s <-> embedding_4,
                                %s <-> embedding_5
                            ) AS similarity
                        FROM person_embeddings
                    )
                    SELECT national_id, person_name, similarity
                    FROM similarities
                    ORDER BY similarity
                    LIMIT 1;
                """
                cursor.execute(query, (compare_vector_str, compare_vector_str, compare_vector_str, compare_vector_str, compare_vector_str))
                result = cursor.fetchone()

                if result[2] <= 1:
                    df.at[idx, 'Matched Name'] = result[1]
                    df.at[idx, 'Matched ID'] = result[0]
                    df.at[idx, 'Similarity'] = float(result[2])
                    df.at[idx, 'Matched'] = True
                    continue
                else:
                    df.at[idx, 'Matched'] = False

        cursor.close()

    conn.close()

    matched_df = df[df['Matched'] == True]
    unmatched_df = df[df['Matched'] == False].drop(columns=['Matched Name', 'Matched ID', 'Similarity'])

    print('matched',flush=True)
    print(matched_df[['Matched Name', 'Matched ID', 'Similarity']].head(5), flush=True)
    print(100*'*', flush=True)
    print('unmatched',flush=True)
    print(unmatched_df, flush=True)
    if not matched_df.empty:
        send_to_queue(matched_df, os.environ['GB_QUEUE'])
    if not unmatched_df.empty:
        send_to_queue(unmatched_df, os.environ['CLUSTER_QUEUE'])

def start_matching():
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=os.environ['EMB_QUEUE'])

    channel.basic_consume(queue=os.environ['EMB_QUEUE'], on_message_callback=compute_matching, auto_ack=True)
    channel.start_consuming()

start_matching()



