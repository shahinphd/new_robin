

import json
import pika
import os
import pandas as pd
import psycopg2
import math

def distance_to_score(distance):
    thr = 0.99
    return 1 / (1 + math.exp(2 * (distance - thr) / 0.28))

def send_to_queue(df, queue_name, ttl = None):
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    if ttl is None:
        channel.queue_declare(queue=queue_name)
    else: 
        channel.queue_declare(queue=queue_name, 
                              arguments={'x-message-ttl':ttl})
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
    df['DB id'] = None
    df['Matched Name'] = None
    df['Matched ID'] = None
    df['Similarity'] = None
    df['Matched'] = False
    df['Face Score Orig'] = None



    SIMILARITY_THRESHOLD = 0.8
    
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
        # recent_match = cursor.fetchall()
        
        if recent_match:



            dis_score =  distance_to_score(recent_match[10])
            # dis_score =  recent_match[10]

            # print("recent_match dis_score", dis_score, recent_match[10], recent_match[9], recent_match[7], flush=True)
            if dis_score >= SIMILARITY_THRESHOLD and recent_match[9] is not None: 
                # print( '*****recent_match[11]*******', recent_match[10], flush=True)
                df.at[idx, 'DB id'] = recent_match[0]
                df.at[idx, 'Matched Name'] = recent_match[9]
                df.at[idx, 'Matched ID'] = recent_match[8]
                df.at[idx, 'Similarity'] = float(dis_score)
                df.at[idx, 'Face Score Orig'] = recent_match[5]
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

            
            if rest_match:
                dis_score =  distance_to_score(rest_match[10])
                # dis_score =  rest_match[10]
                # print("rest_match dis_score", dis_score, rest_match[10], rest_match[9], flush=True)
                if dis_score >= SIMILARITY_THRESHOLD and rest_match[9] is not None:
                    # print( '*****rest_match*******', rest_match[10], flush=True)
                    df.at[idx, 'DB id'] = rest_match[0]
                    df.at[idx, 'Matched Name'] = rest_match[9]
                    df.at[idx, 'Matched ID'] = rest_match[8]
                    df.at[idx, 'Similarity'] = float(dis_score)
                    df.at[idx, 'Face Score Orig'] = rest_match[5]
                    df.at[idx, 'Matched'] = True
                continue

            else:
                # print("No match found for vector", flush=True)
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

                 
                # dis_score =  distance_to_score(result[2])
                dis_score = result[2]

                # print("person_embeddings dis_score", dis_score, result[2], result[1], flush=True)
                if dis_score <= 1.1:
                    df.at[idx, 'Matched Name'] = result[1]
                    df.at[idx, 'Matched ID'] = result[0]
                    df.at[idx, 'Similarity'] = float(dis_score)
                    df.at[idx, 'Matched'] = True
                    continue
                else:
                    df.at[idx, 'Matched'] = False

        cursor.close()

    

    matched_df = df[df['Matched'] == True]
    unmatched_df = df[df['Matched'] == False].drop(columns=['Matched Name', 'Matched ID', 'Similarity'])
    print(100*'*', flush=True)
    print('unmatched',flush=True)
    print(unmatched_df['FacePath'], flush=True)
    print(100*'*', flush=True)
    print('matched',flush=True)
    print(matched_df[['FacePath', 'Matched Name', 'Matched ID', 'Similarity']].head(1), flush=True)
    # print(100*'*', flush=True)
    
    if not matched_df.empty:    
        indxs = list(matched_df.groupby('Matched ID')['Score'].idxmax())
        # print( "indxs", indxs, flush=True)
        for idx, row in matched_df.iterrows():
            # print(row, flush= True)
            if idx in indxs:
                if row['DB id'] is None:
                    with conn.cursor() as cursor:
                        # if os.path.exists(row['FramePath']):
                        newFramePath = row['FramePath'].split(".jpg")[0]+'_'+row['Matched Name']+".jpg"
                        os.rename(row['FramePath'], newFramePath)
                        newFacePath = row['FacePath'].split(".jpg")[0]+'_'+row['Matched Name']  +".jpg" 
                        os.rename(row['FacePath'], newFacePath)
                        # print("FacePath", newFacePath,flush=True)

                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            INSERT INTO detected_people (cam_id, face_path, frame_path, face_embed, face_score, datetime, matched, national_id, person_name, similarity)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s , %s)
                            """,
                            (int(row['CameraID']), newFacePath, newFramePath, row['Feature'], row['Score'], row['DateTime'], row['Matched'], row['Matched ID'], row['Matched Name'], row['Similarity'])
                        )
                    conn.commit()
                    send_to_queue(row, os.environ['PEOPLE_QUEUE'], ttl = 1000)
                else:
                    if row['Score'] > row['Face Score Orig']:
                        with conn.cursor() as cursor:
                            # print("FacePath", row['FacePath'],flush=True)
                            # print(row, flush=True)
                            # if os.path.exists(row['FramePath']):
                            newFramePath = row['FramePath']+'_'+row['Matched Name']
                            os.rename(row['FramePath'], newFramePath)
                            # if os.path.exists(row['FacePath']):
                            newFacePath = row['FacePath']+'_'+row['Matched Name']
                            os.rename(row['FacePath'], newFacePath)
                            cursor = conn.cursor()
                            cursor.execute(
                                """
                                UPDATE detected_people SET (cam_id, face_path, frame_path, face_embed, face_score, datetime, matched, national_id, person_name, similarity)
                                = (%s, %s, %s, %s, %s, %s, %s, %s, %s , %s) WHERE id = %s
                                """,
                                (int(row['CameraID']), newFacePath, newFramePath, row['Feature'], row['Score'], row['DateTime'], row['Matched'], row['Matched ID'], row['Matched Name'], row['Similarity'] , row['DB id'])
                            )
                        conn.commit()
                        send_to_queue(row, os.environ['PEOPLE_QUEUE'], ttl = 1000)
            else:
                # if os.path.exists(row['FramePath']):
                os.remove(row['FramePath'])
                # if os.path.exists(row['FacePath']): 
                print("matched", row['FacePath'], flush=True )   
                os.remove(row['FacePath'])  
    conn.close()
    if not unmatched_df.empty:
        send_to_queue(unmatched_df, os.environ['CLUSTER_QUEUE'])
    # if not matched_df.empty:
    #     send_to_queue(matched_df, os.environ['PEOPLE_QUEUE'])

def start_matching():
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=os.environ['EMB_QUEUE'])
    channel.basic_consume(queue=os.environ['EMB_QUEUE'], on_message_callback=compute_matching, auto_ack=True)
    channel.start_consuming()

start_matching()