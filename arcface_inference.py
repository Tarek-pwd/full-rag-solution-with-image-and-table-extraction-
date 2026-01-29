import faiss
import pickle
import insightface
import numpy as np
import cv2

# target_path= 'no.png'
index = faiss.read_index('faces.index')
with open('id_to_name.pkl' ,'rb') as my_dict:
    id_to_img = pickle.load(my_dict)
print("got the dict " , id_to_img)

model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))

def faces_model(target_image):
    faces = model.get(target_image)
    final_string  = ''
    if faces:
        print("faces detevted !")
        for i in range(len(faces)):
            embedding = faces[i].embedding.astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.reshape(1, -1)
            D,I = index.search(embedding,k=1)
            if D[0][0] > 0.6:
                final_string += id_to_img[int(I[0][0])]    
                if (i < len(faces)-1):
                    final_string+= ' and '
        final_string = 'Image containing ' + final_string
        print("final string " , final_string)
        return (1,final_string)

    else:
        return (0,"no faces detected")
        ## return something to allow for here 
        













