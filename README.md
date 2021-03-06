# Object detector



YOLO 와 OpenCv를 이용해서 물체를 찾고 분류한다.

## 필수사항

- opencv
- numpy

`pip install numpy opencv-python`

#### 학습된 yolo weight 다운

`$ git clone https://github.com/HeeTop/object-detector.git `

`$ cd object-detector`

 `$ wget https://pjreddie.com/media/files/yolov3.weights`



#### Command format



`$ python make_video.py —save <save path>` 

`$ python yolo_opencv.py --image <image path> --save <save path> `

`$ python yolo_opencv.py --video <video path> --save <save path>`



#### Sample Output



##### Image

`$ python yolo_opencv.py --image data/dog.jpg`

Input image

 ![alt text](https://github.com/HeeTop/object-detector/blob/master/data/dog.jpg)

Output image

![alt text](https://github.com/HeeTop/object-detector/blob/master/data/object-detection.jpg)



##### Video

`$ python yolo_opencv.py --video data/demo.py`

Input video

![demo](https://github.com/HeeTop/object-detector/blob/master/data/demo.gif)

Output video

![demo](https://github.com/HeeTop/object-detector/blob/master/data/output.gif)



​      



​     

