# Object detector



YOLO 와 OpenCv를 이용해서 물체를 찾고 분류한다.

## 필수 사항

- opencv
- numpy

`pip install numpy opencv-python`

#### 학습된 yolo weight 다운받아 줌

 `$ wget https://pjreddie.com/media/files/yolov3.weights`



#### Command format

`$ python yolo_opencv.py --image <image path> --save <save path>`

`$ python yolo_opencv.py --video <video path> --save <save path>`



#### Sample Output

`$ python yolo_opencv.py --image dog.jpg`

input image

 ![alt text](object-detector/data/dog.jpg)

Output image

![alt text](object-detector/data/object-detection.jpg)





​      



​     

