
import numpy as np
import json
import pika
import os
import pandas as pd
import networkx as nx
import psycopg2
from datetime import datetime


def send_to_queue(df, queue_name):
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    message = df.to_json()
    channel.basic_publish(exchange='', routing_key=queue_name, body=message)


def cosine_similarity(vectors):
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_vectors, norm_vectors.T)
    return similarity_matrix

def cluster(df):
    features = list(df['Feature'])
    cos_threshold = float(os.environ['CLUSTER_THRESHOLD'])
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
        high_comp = -1
        comp = list(component)
        for i in comp:
            score = df['Score'][i]
            if high_score < score:
                high_score = score
                if high_comp >= 0:
                    os.remove(df['FramePath'][high_comp])
                    print("cluster", df['FacePath'][high_comp], flush=True)    
                    os.remove(df['FacePath'][high_comp])
                high_comp = i
            else:   
                os.remove(df['FramePath'][i])
                os.remove(df['FacePath'][i])
        idx.append(high_comp)
    df = df.iloc[idx]
    return df

def compute_clustering(channel, method, properties, body):

    body = json.loads(body)
    df = pd.DataFrame.from_dict(body)
    later_time =  datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    if df.shape[0] > 1 :
        df = cluster(df)

    GLCLST = pd.concat([GLCLST, df],ignore_index = True)

    if ((later_time - now_time).seconds)/60 > 1:  
        GLCLST = cluster(GLCLST)
        conn = psycopg2.connect(
        dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT']
    )
        for _, row in gf.iterrows():
            print(row)
            with conn.cursor() as cursor:
                cursor.execute( 
                    """
                    INSERT INTO detected_people (cam_id, face_path, frame_path, face_embed, face_score, datetime, matched)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (int(row['CameraID']), row['FacePath'], row['FramePath'], row['Feature'], row['Score'], row['DateTime'], row['Matched'])
                )

            conn.commit()
            send_to_queue(row, os.environ['PEOPLE_QUEUE'])
        conn.close()
        GLCLST = pd.DataFrame(None)
        now_time =  datetime.today().strftime('%Y-%m-%d_%H:%M:%S')



def start_clustering():
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=os.environ['CLUSTER_QUEUE'])

    channel.basic_consume(queue=os.environ['CLUSTER_QUEUE'], on_message_callback=compute_clustering, auto_ack=True)
    channel.start_consuming()

global GLCLST
GLCLST = pd.DataFrame(None)
global now_time
now_time =  datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
start_clustering()

    