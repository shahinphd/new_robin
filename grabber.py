import base64
import cv2
import numpy as np
import pandas as pd
import torch
import ffmpegio
from model_detection.models.mtcnn import MTCNN
from call_function_with_timeout import SetTimeout
from fastapi import FastAPI
from pydantic import BaseModel
import gc
import torch.nn.functional as F
from time import time
from torchvision import transforms
import math
import PIL
from fastapi.responses import HTMLResponse
import networkx as nx
from scipy.spatial.distance import cdist


def get_ffmpeg(video_file, fps, duration):
    frames = []
    with ffmpegio.open(video_file, 'rv', t=duration, r=fps) as f:
        for frame in f:
            frames.append(frame[0])
    return frames

# MagFace Functions
def load_pretrained_model(architecture='mag'):
    ckpt = torch.load('models/face.mag.unpg.pt', map_location=device)  # load checkpoint
    model_embed = ckpt['backbone'].to(device)
    return model_embed



def to_input(pil_rgb_image):
    img = PIL.Image.fromarray(pil_rgb_image).convert("RGB")
    R, G, B = img.split()
    img = PIL.Image.merge("RGB", (B, G, R))
    data_transform = transforms.Compose([
                    transforms.Resize([112,112]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    img = data_transform(img)
    return img

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

def cosine_similarity(vectors):
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_vectors, norm_vectors.T)
    return similarity_matrix

def get_features(vid_images = None, clustering_threshold=0.3,
                       fps = 2, duration=10, device = 'cpu'): 
    total_frames = fps * duration
    idf = pd.DataFrame(None)
    faces_score = []
#     faces_prob = []

    for dr in range(0, int(total_frames)+1,20):
        images = vid_images[dr:dr+20]
        faces_fps = []
        faces_box = []
        faces_mark = []
        faces = []
        f_score = []
        with torch.no_grad():
            frames_face, face_prob, boxes, landmarks = mtcnn(images, return_prob=True)
        for j, frames in enumerate(zip(frames_face, boxes, face_prob, landmarks)):
            if frames[0] is not None:
                for l in range(len(frames[0])):
                    f = frames[0][l]
                    b = frames[1][l]
#                     p = frames[2][l]
                    m = frames[3][l]
                    st = time()
                    pose = find_pose(m.transpose())
                    ff = (f.permute(1, 2, 0).cpu().numpy()).astype("uint8")
                    fff = np.array(cv2.cvtColor(ff, cv2.COLOR_BGR2GRAY))
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
                        bb = bbox(margin, b, image_size, images[0])
                        faces_mark.append(pose)  
                        ff = to_input(ff)
                        faces.append(ff)
                        faces_fps.append(dr+j)
                        faces_box.append(bbox(margin, b, image_size, images[0]))
#                         faces_prob.append(p)
                        faces_score.append(face_score)
                        f_score.append(face_score)
        if len(faces) != 0:
            faces = torch.stack(faces, dim=0)
            faces = faces.to(device)
        else:
            continue
        with torch.no_grad():
            features = model_magface(faces)
            features = F.normalize(features)
        features = features.detach().cpu().numpy()
        data = {'Video_Number':np.full(len(faces_fps),video_number),
                'Frame_Number':faces_fps,'Angles':faces_mark ,
                'Features':list(features),'Bounding_Box':faces_box, 'Score':f_score}
        idf = pd.concat([idf,pd.DataFrame(data)],ignore_index = True)
        del(pose, ff, fff, boxes, landmarks, data)
        del(face_score, pitch_yaw_weighted, sharp2)
        del(faces, faces_box, faces_fps, f, b, m)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()  
    if idf.shape[0] == 0:
        return None    
    elif idf.shape[0] == 1:
        return idf
    
    # Clustering
    features = list(idf['Features'])
    st = time()
    cos_sim = cosine_similarity(features)
    cos_threshold = clustering_threshold
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
            score = idf['Score'][i]
            if high_score < score:
                high_score = score
                high_comp = i
        idx.append(high_comp)
    idf = idf.iloc[idx]
    del(images, features, faces_score)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    return idf

app = FastAPI()

class Params(BaseModel):
    vid_dir: str = None
    clustering_threshold: float = 0.3
    fps: int = 5
    duration: int = 10
    min_height: int = 70
    min_width: int = 60
    
video_number = 1
image_size = 112
margin = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
    torch.cuda.empty_cache()
    
mtcnn = MTCNN(image_size=image_size, margin=margin,min_face_size=100, thresholds=[.95, .95, .95],
              keep_all=True, post_process=False,device=device)
model_magface = load_pretrained_model('mag')

@app.post("/vid/")
async def get_vid(params : Params):
    
    st0 = time()
    func_with_timeout = SetTimeout(get_ffmpeg, timeout=30)
    _, is_timeout, _, images = func_with_timeout(params.vid_dir, params.fps, params.duration)
    if is_timeout:
        html_content = "Time Out!"
        return HTMLResponse(content=html_content, status_code=500)
    if images is None:
        html_content = "No valid Input"
        return HTMLResponse(content=html_content, status_code=400)
     
    images = [np.asarray(i) for i in images]
    images = np.asarray(images)
    images = torch.from_numpy(images)
    en = time()
    print("get video :", en - st0)  
    st = time()
    # dff = get_features(vid_images = images, clustering_threshold=params.clustering_threshold,
    #                        fps = params.fps, duration=params.duration, device = device)
    try:
        dff = get_features(vid_images = images, clustering_threshold=params.clustering_threshold,
                           fps = params.fps, duration=params.duration, device = device)
        if dff is None:
            html_content = "No Face Detected!"
            return HTMLResponse(content=html_content, status_code=202)
    except:
        html_content = "No valid Input"
        return HTMLResponse(content=html_content, status_code=400)
    
    en = time()
    print("get featured :", en - st)  
    st = time()
    im_base = []
    dff.reset_index(inplace=True, drop=True)
    for i in range(len(dff)):
        fce = (np.array(images[dff.iloc[i][1]]
                        [dff['Bounding_Box'][i][1]:dff['Bounding_Box'][i][3],
                         dff['Bounding_Box'][i][0]:dff['Bounding_Box'][i][2]]))
        if fce.shape[0] < params.min_height or fce.shape[1] < params.min_width:
            continue
        img = cv2.imencode('.jpg', fce[:, :, ::-1])[1].tobytes()
        img_encode = base64.b64encode(img)
        my_json = img_encode.decode('utf8').replace("'", '"')
        im_base.append({"base64":my_json, "feature":(dff['Features'][i]/np.linalg.norm(dff['Features'][i])).tolist()}) 
    en = time()
    print("to base64 :", en - st)       
    del(dff,fce,images)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache() 
        gc.collect()
    print("total: ", time() - st0)
    
    return im_base