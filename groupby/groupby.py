import json
import pika
import os
import pandas as pd
import psycopg2



def compute_groupby(channel, method, properties, body):

    conn = psycopg2.connect(
        dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT']
    )
    body = json.loads(body)
    df = pd.DataFrame.from_dict(body)
    indxs = list(df.groupby('Matched ID')['Score'].idxmax())
    # print('idxs******************',indxs,flush=True)
    for idx, row in df.iterrows():
        if idx in indxs:
            with conn.cursor() as cursor:
                if os.path.exists(row['FramePath']):
                    newFramePath = row['FramePath']+'_'+row['Matched Name']
                    os.rename(row['FramePath'], newFramePath)
                if os.path.exists(row['FacePath']):  
                    newFacePath = row['FacePath']+'_'+row['Matched Name']   
                    os.rename(row['FacePath'], newFacePath)

                    
                cursor = conn.cursor()         
                cursor.execute(
                    """
                    INSERT INTO detected_people (cam_id, face_path, frame_path, face_embed, face_score, datetime, matched, national_id, person_name, similarity)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s , %s)
                    """,
                    (int(row['CameraID']), newFacePath, newFramePath, row['Feature'], row['Score'], row['DateTime'], row['Matched'], row['Matched ID'], row['Matched Name'], row['Similarity'])
                )
            conn.commit()
        else:
            if os.path.exists(row['FramePath']):
                os.remove(row['FramePath'])
            if os.path.exists(row['FacePath']):    
                os.remove(row['FacePath'])  

           
    print("groupby", df.head(5), flush=True)
    conn.close()


def start_groupby():
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=os.environ['GB_QUEUE'])

    channel.basic_consume(queue=os.environ['GB_QUEUE'], on_message_callback=compute_groupby, auto_ack=True)
    channel.start_consuming()


start_groupby()
