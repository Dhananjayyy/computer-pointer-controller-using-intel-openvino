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


## How to run (Demo):
### Method 1: (For Windows users only)

- Double click on the `script.bat` file or open cmd in the project root folder and run `script.bat`
- `Prompt: Initializing OpenVINO environment
- If your system have successfully installed OpenVINO environment and the requirement, it will be initialised.

- `Prompt: This project requires virtual environment, proceed to create? (Y/[N])`
- Pressing 'y' will create the virtual environment in your current directory. Press 'n' if it already exists.

- `Prompt: Download required dependancies in your virtual environment? (Y/[N])`
- This project requires certain packages to be installed in the virtual env to effectively run it.
- It is stored in `requirements.txt` file.
- Pressing 'y' will download them. PRessing 'n' will skip this step.

- `Prompt: Proceed to download the required models? (Y/[N])`
- This project requires four models from ithe model downloader: `Gaze Detection Model, Face Detection Model, Head Pose Estimation Model, Facial Landmarks Detection Model`
- Press 'y' to proceed and 'n' to skip.

- `Prompt: Here's your script to run the project:
- You will be displayed a generated script to run this project.

- `Proceed to execute the generated script? (Y/[N])`
- Press 'y' to execute the above script and run the project.
- If above steps are complted successfully, the project will start


### Method 2: (For Windows users only)
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
python {path to main.py file} -fdm {path to the xml file of face detection model} -ldm {path to the xml file of landmark detection model} -hpm {path to the xml file of head pose estimation model} -gem {path to the xml file of gaze estimation model} -ip {path to the video file demo.mp4} -flags fdm ldm hpm gem
```
## Documentation: 
Command Line Argument Information:
- fdm : Path of xml file of face detection model
- ldm : Path of xml file of landmark regression model
- hpm : Path of xml file of Head Pose Estimation model
- gem : Path of xml file of Gaze Estimation model
- ip : Path of input Video file or cam for Webcam
- flags (Optional): To preview video in separate window you need to Specify flag from fdm, ldm, hpm, gem
- prob (Optional): To specify confidence threshold for face detection.  
- d (Optional): Specify Device for inference, the device can be CPU, GPU, VPU, FPGA, MYRIAD
 
### Project Structure:

```
C:.
│   README.md
│   requirements.txt
│   script.bat
│
├───files
│       demo.mp4
│       fp16fps.png
│       fp16inf.png
│       fp16load.png
│       fp32fps.png
│       fp32inf.png
│       fp32load.png
│       int8fps.png
│       int8inf.png
│       int8load.png
│
├───models
│   └───intel
│       ├───face-detection-adas-binary-0001
│       │   └───FP32-INT1
│       │           face-detection-adas-binary-0001.bin
│       │           face-detection-adas-binary-0001.xml
│       │
│       ├───gaze-estimation-adas-0002
│       │   ├───FP16
│       │   │       gaze-estimation-adas-0002.bin
│       │   │       gaze-estimation-adas-0002.xml
│       │   │
│       │   ├───FP32
│       │   │       gaze-estimation-adas-0002.bin
│       │   │       gaze-estimation-adas-0002.xml
│       │   │
│       │   └───FP32-INT8
│       │           gaze-estimation-adas-0002.bin
│       │           gaze-estimation-adas-0002.xml
│       │
│       ├───head-pose-estimation-adas-0001
│       │   ├───FP16
│       │   │       head-pose-estimation-adas-0001.bin
│       │   │       head-pose-estimation-adas-0001.xml
│       │   │
│       │   ├───FP32
│       │   │       head-pose-estimation-adas-0001.bin
│       │   │       head-pose-estimation-adas-0001.xml
│       │   │
│       │   └───FP32-INT8
│       │           head-pose-estimation-adas-0001.bin
│       │           head-pose-estimation-adas-0001.xml
│       │
│       └───landmarks-regression-retail-0009
│           ├───FP16
│           │       landmarks-regression-retail-0009.bin
│           │       landmarks-regression-retail-0009.xml
│           │
│           ├───FP32
│           │       landmarks-regression-retail-0009.bin
│           │       landmarks-regression-retail-0009.xml
│           │
│           └───FP32-INT8
│                   landmarks-regression-retail-0009.bin
│                   landmarks-regression-retail-0009.xml
│
├───openvino_env
└───src
        face_detection.py
        facial_landmark_detection.py
        gaze_estimation.py
        head_pose_estimation.py
        input_feeder.py
        main.py
        mouse_controller.py
```

The `src` folder has 4 model class files which are modularised and other required files
* `face_detection_model.py`
* `gaze_estimation_model.py`
* `landmark_detection_model.py`
* `head_pose_estimation_model.py`

* `main.py` Run complete pipeline of the total project.
* `mouse_controller.py` contains code to move mouse curser pointer based on mouse coordinates.
* `input_feeder.py` contains code to load local video/webcam feed

The `bin` folder contains the demo file and benchmark images.

## Benchmarks

I have checked Inference Time, Model Loading Time, and Frames Per Second on different machines.
I have run the model in 5 diffrent hardware named:
IEI Mustang F100-A10 FPGA
Intel Xeon E3-1268L v5 CPU
Intel Atom x7-E3950 UP2 GPU
Intel Core i5-6500TE CPU
Intel Core i5-6500TE GPU

---

### INT8
#### Inference Time
![](/files/int8inf.png)
#### Loading Time
![](/files/int8load.png)
#### FPS
![](/files/int8fps.png)

---

### FP16
#### Inference Time
![](/files/fp16inf.png)
#### Loading Time
![](/files/fp16load.png)
#### FPS
![](/files/fp16fps.png)

---

### FP32
#### Inference Time
![](/files/fp32inf.png)
#### Loading Time
![](/files/fp32load.png)
#### FPS
![](/files/fp16fps.png)

---

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
