from pdf2image import convert_from_path
import cv2
import numpy as np
import layoutparser as lp
from ultralytics import YOLO
import os
import pytesseract 
from arcface_inference import faces_model
from Blip import caption_image
from pp import table_extraction
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from PIL import Image
from faiss_it import create_FAISS_index
# {0: 'Caption', 1: 'Footnote', 10: 'Title', 2: 'Formula', 3: 'List-item', 4: 'Page-footer', 5: 'Page-header', 6: 'Picture', 7: 'Section-header', 8: 'Table', 9: 'Text'}
ocr_compatible = [7,9]
ocr_comp_set = set()
for elem in ocr_compatible:
    ocr_comp_set.add(elem)



model_path = '/Users/tarekradwan/Downloads/yolo-doclaynet.pt'
model = YOLO(model_path)

os.makedirs('generated/images',exist_ok=True)
image_directory = 'generated/images'


color_map = {
    0: (255, 0, 0),     # Blueq
    1: (0, 255, 0),     # Green
    2: (0, 0, 255),     # Red
    3: (255, 255, 0),   # Cyan
    4: (255, 0, 255),   # Magenta
    5: (0, 255, 255),   # Yellow
    6: (128, 0, 128),   # Purple
    7: (0, 128, 255),   # Orange
    8: (128, 128, 128), # Gray
    9: (0, 0, 0),       # Black
}

chunks_array = []
img_id = 0
last_header = ''
current_text_chunk = defaultdict(list)
last_embedding = np.zeros(384)


def save_img(img, image_path): ## fucntion to save image locally 
    image = Image.fromarray(img, 'RGB')
    image_path = os.path.join(image_directory,image_path)
    image.save(image_path)
    print("image saved to path ", image_path)

def safe_ocr(img):
    if img is None:
        return ""

    if img.size == 0:
        return ""

    h, w = img.shape[:2]
    if h < 10 or w < 10:
        return ""

    # Force contiguous memory (PIL loves this)
    img = np.ascontiguousarray(img)

    try:
        ocr_res = pytesseract.image_to_string(img)
        print('ocr_res is' , ocr_res)
        return ocr_res
    
    except Exception as e:
        print("OCR failed:", e)
        return ""


def find_page_order(results,pg_num,image,results_map):
    global chunks_array
    global img_id
    h,w,_ = image.shape
    text_array = []
    print("currently at page  >> ", pg_num)
    for box in results[0].boxes:
        description = ''
        box_cls = int(box.cls[0].item())
        x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
        cropped_portion =  image[y1:y2 , x1:x2]
        #### check only if text is present there  --- could be section header too

        print("current element is -- : ", results_map[box_cls])
        if box_cls in ocr_comp_set:
            result = safe_ocr(cropped_portion)
            label = results_map[box_cls]
            midpoint = ((x1 + x2)/2,(y1+y2/2))
            text_element = [result,label,midpoint]
            text_array.append(text_element)
        elif box_cls in (6,8):
            caption = caption_image(cropped_portion)
            part_above = image[max(0,y1-200):y1,:]
            text_above = safe_ocr(part_above)
            as_arr = text_above.split('\n')
            print("above" , as_arr)
            text_above = pick_available(as_arr,0)
            part_below = image[y2:min(h-5,y2+200),:]
            text_below = safe_ocr(part_below)
            as_arr = text_below.split('\n')
            print("below" , as_arr)
            text_below = pick_available(as_arr,1)
            # above_below = [part_above,part_below]
            # for elem in above_below:
            #     if elem.size == 0:
            #         continue
            #     while True:
            #         cv2.imshow("box window",elem)
            #         k = cv2.waitKey(1) & 0xFF
            #         if k == 27 or k == ord('q'):
            #             break
            
            if box_cls == 6: ## image detected  ---- 'pick some text below and above image fo
                result = faces_model(cropped_portion)
                if result[0]:
                    description += result[1]
                    description += '\n'
                description += caption
                image_path = 'image' + str(img_id) + '.png'
                chunk = {
                    'type': 'image',
                    'page number' : pg_num,
                    'description' : description ,
                    'image_path' : image_path
                }
                save_img(cropped_portion,image_path)
                img_id +=1

            elif box_cls == 8: ## table
                table_name = caption
                table_content = table_extraction(cropped_portion)
                chunk = {
                    'type': 'table',
                    'page number' : pg_num,
                    'description' : table_name,
                    'table content': table_content
                }
            if text_above:
                chunk['text_above'] = text_above 
            if text_below:
                chunk['text_below'] = text_below

            print("appending chunk >> ", chunk)

            chunks_array.append(chunk)




    ######## processing all extracted text and section headers  ########
    ##print("got text_array :: ", text_array)
    sorted_coords = sorted(text_array, key = lambda x :x[2][1])
    ##print("sorted_coords >> ", sorted_coords)
    if sorted_coords:
        last_x = sorted_coords[0][2][0] ## forst met x coordinate
        x_thresold = 400
        for i in range(1,len(sorted_coords)):
            elem = sorted_coords[i]
            current_xc = elem[2][0]
            if (current_xc - last_x) > x_thresold:
                return  (text_array,1)
            last_x = current_xc
    return (text_array,0) 

def pick_available(text_arr, flg):
    iterable = text_arr if flg else reversed(text_arr)
    for elem in iterable:
        if elem:
            return elem
    return ""

        

def sort_double_splitter(res,page_num):
    sequential_array = []
    #print("should get the final sorted double splitter in here!")
    sorted_with_x = sorted( res, key = lambda x : x[2][0])
    x_threshold = 400

    for i in range(1,len(sorted_with_x)):
        elem = sorted_with_x[i]
        difference = elem[2][0] - sorted_with_x[i-1][2][0]
        if difference > x_threshold:
            break

    #print("loop exit with index >>  " , i)
    left_column = sorted(sorted_with_x[:i],key = lambda x :x[2][1] )
    right_column = sorted(sorted_with_x[i:] , key = lambda x:x[2][1])
    ##print("left column" , left_column)
    ##print("right column", right_column)
    final_sorted_text = left_column + right_column

    for elem in final_sorted_text:
        sequential_array.append(elem[:2])
    text_sequential_chunking(sequential_array,page_num)

    ##print("START " * 10)
    ##print(sequential_array)
    ##print("END " *   50)

def text_sequential_chunking(final_sorted_text,current_page):
    similarity_threshold = 0.5
    global last_header
    global current_text_chunk
    global last_embedding
    for elem in final_sorted_text: 
        elem_label = elem[1]
        elem_content = elem[0]
        if 'header' in elem_label:
            last_header = elem_content
            #print("current header is now " , last_header)
            ### append the current text chunk  ---- set the current chunk to empty and vector to init ### 
            if current_text_chunk['text']:
                print("[H] APPENDING TEXT CHUNK [H]::" , current_text_chunk)
                chunks_array.append(current_text_chunk)
            #print(current_text_chunk)
            init_current_chunk(np.zeros(384))
            continue
        ### if text element was  found --- need tp get the embedding§s #### 
        print("Text found")
        current_text_embedding = embedding_model.encode(elem_content)
        similarity_with_last = cosine_similarity ([current_text_embedding],[last_embedding])[0][0]
       # print(f"similarity of {elem_content} with prev is {similarity_with_last} " )
        if similarity_with_last > similarity_threshold:
            print("merging chunks")
            last_embedding = np.vstack([current_text_embedding,last_embedding]).mean(axis=0)
        else:
            if current_text_chunk['text']:
                chunks_array.append(current_text_chunk)
                print("starting new chunk -- and appending previpous")
                print('[S] APPENDNING TEXT CHUNK [S]::',current_text_chunk)
            init_current_chunk(current_text_embedding)
        current_text_chunk['pages'].append(current_page)
        current_text_chunk['text'].append(elem_content)
        current_text_chunk['header'] = last_header
    # chunks_array.append(current_text_chunk)


def init_current_chunk(curr_emb):
    global current_text_chunk
    global last_embedding
    current_text_chunk = defaultdict(list)
    last_embedding = curr_emb


def ingest_document(pdf_path,progress_callback = None) :
    progress = 0
    pages = convert_from_path(pdf_path)
    total_pages = len(pages)
    results_map = {}
    for i, page in enumerate(pages):
        image = np.array(page)
        print(f"\nPAGE {i}")
        results = model.predict(image)

        if i == 0: 
            results_map = results[0].names
            print("INITIALED RESULTS MAP >>" , results_map)

        res,flg = find_page_order(results,i,image,results_map)
        if flg :
            print('double split page')
            sort_double_splitter(res,i)
        else:
            print('single split page')
            sorted_text = sorted(res, key = lambda x: x[2][1])
            text_sequential_chunking(sorted_text,i)

        if progress_callback:
            progress = int(((i + 1) / total_pages) * 100)
            progress_callback(progress)


    chunks_array.append(current_text_chunk)
    create_FAISS_index(chunks_array)




# ingest_document('gg.pdf')

#  cv2.putText(image, label, (top_left[0], top_left[1] - 10 - baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
# def display_results(results,image,pg_num):

#     # print(f" FUNCTION CALLED PAGE NUM {pg_num} and SECTION HEADER {section_header}")

#     global section_header
#     elements_detected = []
#     for box in results[0].boxes:
#         detected_element = {}
#         description = ''
#         box_cls = int(box.cls[0].item())
#         box_conf = float(box.conf[0].item())
#         x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
#         ##print("x1,x2,y1,y2" , x1,x2,y1,y2)
#         color = color_map[box_cls]
#         label = results_map[box_cls]
#         cropped_portion =  image[y1:y2 , x1:x2]

#         if box_cls in ocr_comp_set:
#             result = pytesseract.image_to_string(cropped_portion)
#             if box_cls == 7:
#                 print("section headeer detected !! changing glonal variable ", result) ## checking if it is a  a sevtion header
#                 section_header = result
                

#             print(f"text detected {result} and the header is {section_header}"  )
#             description = result
    

#         elif box_cls == 6: ## image detected 
#             result = faces_model(cropped_portion)
#             if result[0]:
#                 description += result[1]
#                 description += '\n'
#             description += caption_image(cropped_portion)
#             ##print("description >> " , description )
            
#         elif box_cls == 8:
#             print('table_found on page  >> ')
#             description = table_extraction(cropped_portion)


#         detected_element['page number'] = pg_num
#         detected_element['section header'] = section_header
#         detected_element['type'] = label
#         detected_element['description'] = description
 
#         cv2.rectangle(image, (x1,y1),(x2,y2),color,4)
#         cv2.putText(image,label + " " + str(box_conf),(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,cv2.LINE_AA)
#         elements_detected.append(label)

        

#     print("detectred : : : : :  " , elements_detected)
#     return image

