@echo off
echo -------------------------------------------------------------------------------
timeout 1 >nul
set mypath=%cd%
:PROMPT
ECHO Initialising OpenVINO environment
timeout 1 >nul
set ov="C:\Program Files (x86)\IntelSWTools\openvino\bin\"
CALL %ov%\setupvars.bat
:no
echo -------------------------------------------------------------------------------
timeout 1 >nul
:PROMPT
SET /P var=This project requires virtual environment, proceed to create? (Y/[N])
if '%var%'=='N' goto no
if '%var%'=='n' goto no
echo Creating virtual environment
python -m venv openvino-env
:no
set v=%mypath%\openvino-env\Scripts\
CALL %v%\activate
echo Virtual environment is initialised at %v%
echo -------------------------------------------------------------------------------
timeout 1 >nul
:PROMPT
SET /P var=Download required dependancies in your virtual environment? (Y/[N])
if '%var%'=='N' goto end
if '%var%'=='n' goto end
pip install -r requirements.txt
:end
if not exist %cd%/models mkdir %cd%/models
cd "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\"
echo -------------------------------------------------------------------------------
timeout 2 >nul
:PROMPT
SET /P var=Proceed to download the required models? (Y/[N])
if '%var%'=='N' goto skip
if '%var%'=='n' goto skip
echo Downloading...
python downloader.py --name face-detection-adas-binary-0001 -o %mypath%/models
python downloader.py --name gaze-estimation-adas-0002 -o %mypath%/models
python downloader.py --name landmarks-regression-retail-0009 -o %mypath%/models
python downloader.py --name head-pose-estimation-adas-0001 -o %mypath%/models
echo "Download complete"
:skip
set MainFilePath=%mypath%\src\main.py
set FaceDetectionPath=%mypath%\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml
set LandmarkDetectionPath=%mypath%\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009.xml
set HeadPoseEstimationPath=%mypath%\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml
set GazeEstimationPath=%mypath%\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002.xml
set edgescript=python %MainFilePath% -fdm %FaceDetectionPath% -ldm %LandmarkDetectionPath% -hpm %HeadPoseEstimationPath% -gem %GazeEstimationPath% -ip %mypath%\files\demo.mp4 -flags fdm ldm hpm gem
echo.
timeout 1 >nul
echo -------------------------------------------------------------------------------
echo Here is your script to run the project:
echo -------------------------------------------------------------------------------
echo %edgescript%
echo -------------------------------------------------------------------------------
cd %mypath%
timeout 1 >nul
:PROMPT
SET /P var=Proceed to execute the generated script? (Y/[N])
timeout 1 >nul
if '%var%'=='N' goto hide
if '%var%'=='n' goto hide
echo "Predd ESC key to exit"
%edgescript%
:hide
echo -------------------------------------------------------------------------------
