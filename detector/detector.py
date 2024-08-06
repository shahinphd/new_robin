import pika
import cv2
import os
import torch
import pandas as pd
from datetime import datetime
import numpy as np
from model_detection.models.mtcnn import MTCNN
import gc
import base64
import json
# import pytz
import math
import shutil
from collections import defaultdict

def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size
    
def bbox(margin, bb, img_size, img):
    margin = [
        margin * (bb[2] - bb[0]) / (img_size - margin),
        margin * (bb[3] - bb[1]) / (img_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(bb[0] - margin[0] / 2, 0)),
        int(max(bb[1] - margin[1] / 2, 0)),
        int(min(bb[2] + margin[0] / 2, raw_image_size[0])),
        int(min(bb[3] + margin[1] / 2, raw_image_size[1])),
    ]
    return box

def find_pose(points):
    LMx = points[0]
    LMy = points[1]
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = (LMy[1] - LMy[0])
    angle = np.arctan(dPy_eyes / dPx_eyes) 
    alpha = np.cos(angle)
    beta = np.sin(angle)
    LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2) 
    LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2
    yaw = (-90+90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    pitch = (-90+90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0
    roll = angle * 180 / np.pi 
    return roll, yaw, pitch

def normal_weight(max_limit, input_value, percent):
    if (max_limit-input_value) < 0:
        input_value = max_limit
    return ((max_limit-input_value)/max_limit)*percent    

def mtcnn_detection(df_images):

    images = df_images["frame"]
    images = [np.asarray(i) for i in images]
    images = np.asarray(images)
    images2 = torch.from_numpy(images)
    idf = pd.DataFrame(None)
    faces_paths = []
    frames_paths = []
    cam_ids = []
    img_bytes = []
    # face_bytes = []
    # frame_bytes = []
    scores = []
    times = []
    # faces = []

    with torch.no_grad():
        # print("images2", len(images2), images2[0].shape, flush=True)
        frames_face, face_prob, boxes, landmarks = mtcnn(images2, return_prob=True)
        for j, frames in enumerate(zip(frames_face, boxes, face_prob, landmarks)):
            if frames[0] is not None:
                for l in range(len(frames[0])):
                    f = frames[0][l]
                    land = frames[3][l]
                    pose = find_pose(land.transpose())
                    ff = (f.permute(1, 2, 0).cpu().numpy()).astype("uint8") 
                    sharp2 = np.mean(cv2.Canny(ff, 50,250))
                    if sharp2 > 10:
                        sharp2 = 10
                    sharp_weighted= (sharp2/10)*0.5 
                    if abs(pose[1]) > 50 or abs(pose[2]) > 40 or abs(abs(pose[1]) - abs(pose[2])) > 25:
                        pitch_yaw_weighted = 0
                    else:
                        pitch_yaw_weighted = normal_weight(40*50*math.pi/4, abs(pose[2]*pose[1] ),0.5)
                    face_score = pitch_yaw_weighted + sharp_weighted
                    if face_score>0.6:
                        # faces.append(ff)
                        scores.append(face_score)
                        # img_str = cv2.imencode('.jpg', ff)[1].tostring()
                        # face_bytes.append(img_str)
                        # img_str = cv2.imencode('.jpg', images[j])[1].tostring()
                        # frame_bytes.append(img_str)

                        # nparr = np.fromstring(STRING_FROM_DATABASE, np.uint8)
                        # img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)


                        _, buffer = cv2.imencode('.jpg', ff)
                        img_encode =  base64.b64encode(buffer).decode('utf-8')
                        img_bytes.append(img_encode)


                        t =  datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
                        times.append(t)
                        name = "/shared/frames/" + t + "_" + str(j) + ".jpg"
                        
                        cv2.imwrite(name,cv2.cvtColor(images[j], cv2.COLOR_BGR2RGB))
                        frames_paths.append(name)
                        cam_ids.append(df_images["cam_id"][j])
                        name = "/shared/faces/" + t + "_" + str(j) + ".jpg"
                        cv2.imwrite(name, cv2.cvtColor(ff, cv2.COLOR_BGR2RGB))
                        faces_paths.append(name)

    # data = {'CameraID':cam_ids ,'FaceImg':img_bytes, "FacePath":faces_paths, "FramePath":frames_paths, "Score": scores, "DateTime": times}
    data = {'CameraID':cam_ids ,'FaceImg':img_bytes, "FacePath":faces_paths, "FramePath":frames_paths, "Score": scores, "DateTime": times}
    idf = pd.concat([idf,pd.DataFrame(data)],ignore_index = True)
    del(images, images2, frames_paths, faces_paths, cam_ids )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()  
    if idf.shape[0] == 0:
        return None    
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    # print("img_bytes", img_bytes)    
    return idf

def detect_faces(channel, method, properties, body, batch = 50):
    # print(body, flush=True)
    if isinstance(body, (bytes, bytearray)):
        body = body.decode('utf8').replace("'", '"')

    body = json.loads(body)
     
    im = base64.b64decode(body["frame"].encode("utf-8"))
    image_byte_array = bytearray(im) 
    np_array = np.asarray(image_byte_array, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    resolution = image.shape
    global buffers
    buffers[resolution].append({"cam_id": body["cam_id"], "frame":image})

    # Assume detect_faces_in_frame is a function that detects faces and returns face images
    if len(buffers[resolution]) == batch:
        df = pd.DataFrame(buffers[resolution])
        faces_df = mtcnn_detection(df)
        # print(len(batch_frames),"&&&&&&&&&&", flush=True)
        buffers[resolution].clear()
        if faces_df is not None:
            # print(faces_df, flush=True)
            facesdf = faces_df.to_json()
            channel.basic_publish(exchange='', routing_key=os.environ['FACE_QUEUE'], body=facesdf)

def start_detector():

    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=os.environ['FRAME_QUEUE'])
    channel.queue_declare(queue=os.environ['FACE_QUEUE'])
    channel.basic_consume(queue=os.environ['FRAME_QUEUE'], on_message_callback=detect_faces, auto_ack=True)
    channel.start_consuming()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
        torch.cuda.empty_cache()
    
mtcnn = MTCNN(image_size=112, margin=20,min_face_size=100, thresholds=[.95, .95, .95],
              keep_all=True, post_process=False,device=device)

global buffers
buffers = defaultdict(list)
if os.path.exists("/shared/frames"):
    shutil.rmtree("/shared/frames")
os.mkdir("/shared/frames")

if os.path.exists("/shared/faces"):
    shutil.rmtree("/shared/faces")
os.mkdir("/shared/faces")
start_detector()
