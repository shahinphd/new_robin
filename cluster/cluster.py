
import numpy as np
import json
import pika
import os
import pandas as pd
import networkx as nx
import psycopg2


def cosine_similarity(vectors):
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_vectors, norm_vectors.T)
    return similarity_matrix

def cluster(df):
    features = list(df['Feature'])
    cos_threshold = os.environ['CLUSTER_THRESHOLD']
    cos_sim = cosine_similarity(features)
    G = nx.Graph()
    vectors = features
    G.add_nodes_from(range(len(vectors)))
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if cos_sim[i, j] > cos_threshold:
                G.add_edge(i, j)

    # Find connected components
    connected_components = list(nx.connected_components(G))
    idx = []
    for component in connected_components:
        high_score = 0
        high_comp = 0
        comp = list(component)
        for i in comp:
            score = df['Score'][i]
            if high_score < score:
                high_score = score
                high_comp = i

            else:   
                if os.path.exists(df['FramePath'][i]):
                    os.remove(df['FramePath'][i])
                if os.path.exists(df['FacePath'][i]):    
                    os.remove(df['FacePath'])[i] 
        idx.append(high_comp)
    df = df.iloc[idx]
    return df

def compute_clustering(channel, method, properties, body):

    body = json.loads(body)
    df = pd.DataFrame.from_dict(body)
    
    if df.shape[0] == 1:
        df = cluster(df)
        
    conn = psycopg2.connect(
    dbname=os.environ['POSTGRES_DB'],
    user=os.environ['POSTGRES_USER'],
    password=os.environ['POSTGRES_PASSWORD'],
    host=os.environ['POSTGRES_HOST'],
    port=os.environ['POSTGRES_PORT']
)

    print("cluster", df.head(5), flush=True)

    for _, row in df.iterrows():
        with conn.cursor() as cursor:
            cursor.execute( 
                """
                INSERT INTO detected_people (cam_id, face_path, frame_path, face_embed, face_score, datetime, matched)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (row['CameraID'], row['FacePath'], row['FramePath'], row['Feature'], row['Score'], row['DateTime'], row['Matched'])
            )

        conn.commit()
    
    conn.close()


def start_clustering():
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=os.environ['CLUSTER_QUEUE'])

    channel.basic_consume(queue=os.environ['CLUSTER_QUEUE'], on_message_callback=compute_clustering, auto_ack=True)
    channel.start_consuming()

# global gf
# gf = pd.DataFrame(None)

start_clustering()

    