# Object-Detection-in-Night-Light-Conditions
This project uses Zero-DCE model to enhance image and Object Detection using YOLOv3.

Download the YOLO weights using download_weights.sh file in cfg folder.

* For images, store the images in test_data/images folder and run the runThisForPic.py and the results will be stored in test_data/results/images.
* For videos, store the images in test_data/videos folder and run the runThisForVideo.py The fps is set to 12, since we tested it on CPU. It can be changed in the code. The results will be stored in test_data/results/videos.
* For live feed, run the runThisForCam.py and the results will be stored in test_data/results/videos.

The original repository of Zero-DCE - https://github.com/Li-Chongyi/Zero-DCE <br/>
The original repository of Object Detection - https://github.com/ayooshkathuria/pytorch-yolo-v3
