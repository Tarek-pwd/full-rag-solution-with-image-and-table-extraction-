import insightface
import numpy as np
import cv2
import os
import pickle
import faiss

id_to_name = {}

index = faiss.IndexFlatIP(512)

model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))

label_mbs = 'Crown Prince Mohammed Bin Salman'
label_king = 'King Salman Bin Abdulaziz'

folder_name = 'MBS_images'
for img_name in os.listdir(folder_name):
    image_path = os.path.join(folder_name, img_name)
    print("reading image >>", img_name)

    image = cv2.imread(image_path)
    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = model.get(image)
    print("faces:", len(faces))

    if len(faces) != 1:
        continue

    embedding = faces[0].embedding.astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    embedding = embedding.reshape(1, -1)

    if img_name.startswith('ks'):
        curr_label = label_king
    else:
        curr_label = label_mbs

    curr_id = index.ntotal
    index.add(embedding)
    id_to_name[curr_id] = curr_label

    print("registered:", curr_label)


faiss.write_index(index, 'faces.index')

with open('id_to_name.pkl', 'wb') as f:
    pickle.dump(id_to_name, f)
