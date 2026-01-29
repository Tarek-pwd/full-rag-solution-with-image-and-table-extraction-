import cv2
import easyocr


reader = easyocr.Reader(['en'], gpu=False)
img = cv2.imread('ocr_image.png')

results = reader.readtext(img,detail=0)
print(results)