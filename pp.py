import json
from paddleocr import TableStructureRecognition
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=False)


# def plot_bbox(img,pts):
#     print("pts >> " , pts)
#     cv2.rectangle(img, pts[0],pts[1],(255,0,0),4)
#     while True:
#         cv2.imshow("box window",img)
#         k = cv2.waitKey(1) & 0xFF
#         if k == 27 or k == ord('q'):
#             break
#     cv2.destroyAllWindows()


def points_to_polylines(points_arr):
    final_arr = [] ## this will contain thr tuples [(),()]
    for i in range(len(points_arr)-1):
        if i%2 ==0:  ## if the index is an even number , put your self and the next neighbor into a tuple 
            tuple = (points_arr[i],points_arr[i+1])
            final_arr.append(tuple)
    return np.array(final_arr)


def polygon_to_box(arr_tuples):
    #print("got " , arr_tuples)
    min_x = min_y =  float('inf')
    max_x = max_y  = float('-inf') 
    for elem in arr_tuples:
        min_x = int(min(min_x,elem[0]))
        min_y = int(min(min_y,elem[1]))
        max_x = int(max(max_x,elem[0]))
        max_y = int(max(max_y,elem[1]))
    return ((min_x,min_y), (max_x,max_y))     


def extract_text(points,img):   ## points are the top left and bottom right 
    h,w,_ = img.shape
    #print("got the points", points)

    top_left = (max(0,points[0][0]), max(0,points[0][1]))
    bottom_right = (min(w-1,points[1][0]),min(h-1,points[1][1]))
    img_cropped = img[top_left[1]:bottom_right[1] , top_left[0]:bottom_right[0] ]
    # while True:
    #     cv2.imshow("box window",img_cropped)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27 or k == ord('q'):
    #         break
    result = reader.readtext(img_cropped,detail = 0)
    #print("extracted >> " , result)
    #print("extracted type ", type(result))

    return ' '.join(result) if result else 'EMPTY CELL FOUND'
    


def parse_structure(html_structure,td_res):
    final_html = ''
    i = 0
    current_pointer = 0 
    while i < len(html_structure):
        #print("currently at element >> " , html_structure[i])
        if '<td' not in html_structure[i] :
            final_html += html_structure[i]
        else: 
            text_to_append = td_res[current_pointer]
            print("text to apped" , text_to_append)
            current_pointer+=1
            elem = '' 
            if html_structure[i] == '<td></td>':
                #print("double td case ")
                elem = '<td>' + text_to_append + '</td>'
                #print("normal td ", elem)
            else:
                while(html_structure[i]!='</td>'): ## append alll until end
                    elem+=html_structure[i]
                    i+=1
                elem = elem + text_to_append + '</td>'
                #print("col - row span td ", elem)
            final_html+=elem
        i+=1
    return final_html

def refine_bbox(img, pts): 
    MAX_SHIFT_RATIO = 0.01
    def refine_boundary(label_name, points_dict, limit):
        #print("currently exploring", label_name)
        #print('-' * 20)

        y_min = points_dict['y_min']
        x_min = points_dict['x_min']
        x_max = points_dict['x_max']
        y_max = points_dict['y_max']
        #print("iunit y max",y_max)
        #print("imahe shape" , h,w)

        point_val = points_dict[label_name]
        std_threshold = 0
        direction_increment = 1

        def get_line(val):
            if label_name == 'y_min':
                return img[val, x_min:x_max]
            elif label_name == 'y_max':
                return img[val, x_min:x_max]
            elif label_name == 'x_min':
                return img[y_min:y_max, val]
            else:  # x_max
                return img[y_min:y_max, val]

        line_interest = np.array(get_line(point_val))
        saved_std = np.std(line_interest)
        saved_val = point_val

        #print(f'visiting {label_name} --- std {saved_std}')
        dir_count_arr = []

        if saved_std > std_threshold:
            #print("std threshold broken >> entering loop")

            for direction in (-1, 1):
                point_val = saved_val
                direction_count = 0
                line_interest_std = saved_std

                while (line_interest_std > std_threshold and 0 <= point_val < limit-1):
                    point_val += direction * direction_increment
                    direction_count += direction
                    max_shift = int((w if 'x' in label_name else h) * MAX_SHIFT_RATIO)

                    if abs(direction_count) > max_shift:
                        direction_count = float('inf')
                        break


                    # if abs(direction_count) > 50:
                    #     direction_count = float('inf')
                    #     break
                    line_interest = np.array(get_line(point_val))
                    line_interest_std = np.std(line_interest)

                dir_count_arr.append(direction_count)

            #print("direction array", dir_count_arr)
            final_inc = min(dir_count_arr, key=abs)
            final_inc = 0 if final_inc == float('inf') else final_inc
            point_val = saved_val + final_inc
        

        return int(point_val)

    h, w, _ = img.shape
    points = [max(0,pts[0][1]), max(0,pts[0][0]), min(pts[1][1],h-1), min(w-1,pts[1][0])]

    label_names = ['y_min', 'x_min', 'y_max', 'x_max']
    points_dict = {}
    for i in range(len(label_names)):
        points_dict[label_names[i]] = points[i]

    final_arr = []
    for elem in label_names:
        limit = w if 'x' in elem else h
        res = refine_boundary(elem, points_dict, limit)
        final_arr.append(res)
    MIN_W = 2
    MIN_H = 2

    x1, y1 = final_arr[1], final_arr[0]
    x2, y2 = final_arr[3], final_arr[2]

    if x2 - x1 < MIN_W:
        x2 = x1 + MIN_W
    if y2 - y1 < MIN_H:
        y2 = y1 + MIN_H
    return [(x1, y1), (x2, y2)]




def table_extraction(img):

    model = TableStructureRecognition(model_name="SLANet_plus")
    output = model.predict(input=img, batch_size=1)
    td_ocr_result_list=[]
    for res in output:
        res.print(json_format=False)
        res.save_to_json("./output/res.json")
    with open('./output/res.json','rb') as f:
        json_file = json.load(f)
        # print("found json" , json_file)
        structure = json_file['structure']
        #print("got the structure >> " ,structure)
        box_cords = json_file['bbox']
        for i in range(len(box_cords)):
            points = polygon_to_box(points_to_polylines(box_cords[i]))
            points_refined = points
            td_ocr_result_list.append(extract_text(points_refined,img))
            #plot_bbox(img.copy(),points_refined)
        final_r = parse_structure(structure,td_ocr_result_list)
        print(final_r)
        return final_r

####   tr  >>>  as it is
#### td >> text (ocr result) contained  inside <td> res </td>
### td with column or row span would have the col or row span as ana attribute  >> <td colspan = '4'> result </td> 