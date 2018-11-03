import cv2
import os
import numpy as np
import argparse

def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] -1 ] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt

ap = argparse.ArgumentParser()
ap.add_argument('-i' ,'--image',required = True,
                help = '이미지 경로 입력')
ap.add_argument('-v' ,'--video',required = False,
                help = '비디오 경로 입력')
ap.add_argument('-c','--cam',required = False,
                help = '캠 모드')


args = ap.parse_args()


image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None
with open("yolov3.txt",'r') as f:
    classes = [line.strip() for line in f.readlines()]

# different classes different color
COLORS = np.random.uniform(0,255,size=(len(classes),3))

# read pre-trained model and config file
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
blob = cv2.dnn.blobFromImage(image, scale, (Width,Height),(0,0,0), True, crop=False)

# set input blob for the network
net.setInput(blob)


outs = net.forward(get_output_layers(net))

# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        # display output image
        cv2.imshow("object detection", image)

        # wait until any key is pressed
        cv2.waitKey()

        # save output image to disk
        cv2.imwrite("object-detection.jpg", image)

        # release resources
        cv2.destroyAllWindows()



#
#
# # base dir
# video_path = os.path.abspath("./find_spot_video1.mp4")
#
#
# # video load
# cap = cv2.VideoCapture(video_path)
#
#
#
# while(cap.isOpened()):
#
#     ret, white_image = cap.read()
#
#
#
#     # Output Part
#     # cv2.imshow("dilate",dilate)
#     cv2.imshow("yolo",white_image)
#     # cv2.imshow("dilate",dilate)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
# cap.release()
# cv2.destroyAllWindows()
