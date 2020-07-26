# Introduction

Use your gaze to control your computer's mouse pointer movement using Intel OpenVINO toolkit. This project is done completely offline. 
Along with Gaze Detection Model, You will need to use the following models too:
* [Gaze Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
* [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

The gaze detection depends on other models such as head-pose-estimation, face-detection, facial-landmarks.

The gaze estimation model requires following inputs:

* The Head Pose
* The Left Eye Image
* The Right Eye Image.

![demo](/bin/output_video.gif)

You will have to coordinate the flow of data from the input, and then amongst the different models and finally to the mouse controller.

![Pipeline](/images/pipeline.png)

## Project Set Up and Installation:

- Download the **[OpenVino Toolkit](https://docs.openvinotoolkit.org/latest/index.html)** for your system with all the prerequisites.

- Clone the Repository using `git clone https://github.com/Dhananjayyy/computer-pointer-controller-using-intel-openvino.git`

- Create Virtual Environment using command `virtualenv venv` in the command prompt. Make sure to install `virualenv` in python.

- Install all the requirements from "requirements.txt" file using `pip install requirements.txt`.

- Download the required models from `OpenVino Zoo` using the commands below.
It will download required models with all the precisions.

`python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002`

`python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001`  

`python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001`

`python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009`


## Demo:

Step1. Open command prompt. Go to Virtual Environment location. Execute below commands.
```
cd venv/Scripts/
activate
```

Step2. Instantiate OpenVino Environment. (This is very important)
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

Step3. Go to the project directory
```
cd path_to_project_directory
```

Step4. Run below commands to execute the project (demo.mp4)
```
python src/main.py -fd model/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -lr model/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009.xml -hp model/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml -ge model/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -flags ff fl fh fg
```
## Documentation: 
Command Line Argument Information:
- fd : Path of xml file of face detection model
- lr : Path of xml file of landmark regression model
- hp : Path of xml file of Head Pose Estimation model
- ge : Path of xml file of Gaze Estimation model
- i : Path of input Video file or cam for Webcam
- flags (Optional): To preview video in separate window you need to Specify flag from ff, fl, fh, fg
- probs (Optional): To specify confidence threshold for face detection. default=0.7 (Range=0-1)  
- d (Optional): Specify Device for inference, the device can be CPU, GPU, VPU, FPGA, MYRIAD
- o : Specify path of output folder where we will store results
 
### Project Structure:


```
│   output_video.mp4
│   README.md
│   requirements.txt
│
├───bin
│       demo.mp4
│
├───images
│       pipeline.png
│
├───model
│   ├───face-detection-adas-binary-0001
│   │
│   ├───gaze-estimation-adas-0002
│   │
│   ├───head-pose-estimation-adas-0001
│   │
│   └───landmarks-regression-retail-0009
│       
│
└───src
        face_detection_model.py
        gaze_estimation_model.py
        head_pose_estimation_model.py
        input_feeder.py
        landmark_detection_model.py
        main.py
        model.py
        mouse_controller.py
        output_video.mp4
```

- models: This folder contains models in IR format downloaded from Openvino Model Zoo.

- src: This folder contains model files, pipeline file(main.py) and other files.

* `model.py` is the model class file which has common property of all the other model files. It is inherited by all the other model files 
This folder has 4 model class files which are modularised.
* `face_detection_model.py`
* `gaze_estimation_model.py`
* `landmark_detection_model.py`
* `head_pose_estimation_model.py`


* `main.py` Run complete pipeline of the total project.
* `mouse_controller.py` contains code to move mouse curser pointer based on mouse coordinates.
* `input_feeder.py` contains code to load local video/webcam feed

- bin: contains the demo file.

## Benchmarks
I have checked Inference Time, Model Loading Time, and Frames Per Second model for FP16, FP32, and FP32-INT8 of all the models except Face Detection Model. Face Detection Model was only available on FP32-INT1 precision. You can use below commands to get results for respective precisions

---

### Inference Time
![](/images/inference_time.png)
![](/images/inference_time_a.png)

---

### Loading Time

![](/images/model_loading_time.png)
![](/images/model_loading_time_a.png)

---

### FPS

![](/images/fps.png)
![](/images/fps_a.png)

---

**Synchronous Inference**

```
Precision = ['FP16', 'FP32', 'FP32-INT8']
Inference Time : [26.7, 26.3, 26.8]
Loading Time : [1.6324511543623587, 1.6325845657216365, 5.63265478215458]
FPS : [2.124521271181955, 2.2142154412457845, 2.18421648855421]
```

**Asynchronous Inference**

```
Precision = ['FP16', 'FP32', 'FP32-INT8']
Inference Time : [23.9, 24.7, 24.0]
Loading Time : [0.75486324251236, 0.695432365158475, 2.78423651987452]
FPS : [2.12548432154545, 2.53214523654542, 2.212545512236584]
```

## Results:
* Loading time: `FP16` has lowest and `FP32-INT8` has highest.
* Inference time: `FP32` give slightly better results.
* FPS: FP32 gave slightle better results
* Asynchronous Inference has little better results in `inference time` and `FPS` over Synchronous Inference

##

### Edge Cases
* Many People Scenario: If multiple peoples are present in the video, it will  give results on one face.
* Head Detection: When there's no one in the frame, it will skip the frame and inform the user.

### Stand Out Suggestions:
* Proper eye/head/gaze movement is advised 
* Lighting: We might improve pre-processing steps to reduce error due to bad lighting conditions.
