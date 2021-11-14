import matplotlib.pylab as plt
import cv2
import numpy as np

def ROI(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def line_vector(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0,0,255), thickness=4)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (320, height),
        (width/2, height/2),
        (width*2/3, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.GaussianBlur(gray_image,(9,9),cv2.BORDER_DEFAULT)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = ROI(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=20,
                            maxLineGap=100)
    image_with_lines = line_vector(image, lines)
    return image_with_lines

capture = cv2.VideoCapture(r'C:\Users\khush\Downloads\Easy_Test.mp4')
while True:
    ret,frame = capture.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('k'):
        break
capture.release()
cv2.destroyAllWindows()
