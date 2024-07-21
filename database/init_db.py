import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import psycopg2
import random

def to_input(pil_rgb_image):
    # img = Image.fromarray(pil_rgb_image).convert("RGB")
    R, G, B = pil_rgb_image.split()
    img = Image.merge("RGB", (B, G, R))
    data_transform = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = data_transform(img)
    return img

# MagFace Functions
def load_pretrained_model():
    ckpt = torch.load('database/models/face.mag.unpg.pt', map_location=device)  # load checkpoint
    model_embed = ckpt['backbone'].to(device)
    return model_embed

# Function to load and transform images
def load_images(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        to_face = to_input(image)
        images.append(to_face)
    return torch.stack(images).to("cuda")

# Function to extract features using the given code
def extract_features(model, imgs_tensor):
    with torch.no_grad():
        features = model(imgs_tensor)
        features = F.normalize(features)
    return features.cpu().numpy().tolist()

# Function to connect to the PostgreSQL database
def connect_to_db():
    conn = psycopg2.connect(
        dbname="regdb",
        user="postgres",
        password="postgres",
        host="172.18.0.2",
        port='5432'
    )
    return conn

# Function to insert features into the database
def insert_to_db(conn, person_name, features, n_id):
    print(type(features[0]))
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO person_embeddings (national_id, person_name, embedding_1, embedding_2, embedding_3, embedding_4, embedding_5)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (n_id, person_name, features[0], features[1], features[2], features[3], features[4])
        )
    conn.commit()

# Main function to process folders and images
def main(root_folder, model):
    conn = connect_to_db()

    num = len(os.listdir(root_folder))
    list_id = random.sample(range(1111111111, 10000000000), num)


    for j, folder_name in enumerate(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            person_name = folder_name  # Use folder name as person_name
            image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:5]
            if len(image_paths) == 5:  # Ensure there are exactly 5 images
                imgs_tensor = load_images(image_paths)
                features = extract_features(model, imgs_tensor)
                insert_to_db(conn, person_name, features, list_id[j])
    conn.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model_magface = load_pretrained_model()
    # Specify the root folder containing your image folders
    root_folder = 'Celebs'
    main(root_folder, model_magface)
