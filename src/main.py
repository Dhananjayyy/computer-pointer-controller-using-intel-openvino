import os
import cv2
import numpy as np
import time
import logging
from face_detection_model import FaceDetection
from head_pose_estimation_model import HeadPoseEstimationModel
from gaze_estimation_model import GazeEstimationModel
from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
from landmark_detection_model import LandmarkDetectionModel

def build_argparser():
    """
    parse commandline argument
    return ArgumentParser object
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--faceDetectionModel", type=str, required=True,
                        help="Specify path of Face Detection Model's .xml file\n")

    parser.add_argument("-lr", "--landmarkRegressionModel", type=str, required=True,
                        help="Specify path of Landmark Regression Model's .xml file\n")

    parser.add_argument("-hp", "--headPoseEstimationModel", type=str, required=True,
                        help="Specify path of Head Pose Estimation Model's .xml file\n")

    parser.add_argument("-ge", "--gazeEstimationModel", type=str, required=True,
                        help="Specify path of Gaze Estimation Model's .xml file\n")

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Specify input video file's path or cam for live webcam\n")

    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify flags from ff, fl, fh, fg like -flags ff fl\n"
                             "ff for faceDetectionModel, fl for landmarkRegressionModel\n"
                             "fh for headPoseEstimationModel, fg for gazeEstimationModel\n")

    parser.add_argument("-prob", "--threshold_prob", required=False, type=float,
                        default=0.7,
                        help="Specify Probability Threshold for Face Detection Model\n")

    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Specify yhe device to be used for inference\n"
                             "It can be CPU, GPU, VPU, FPGA, MYRIAD\n")
    parser.add_argument("-o", '--path_output', default='/results/', type=str)
    return parser


def draw_preview(
        frame, pre_flag, crop_img, left_eye, right_eye,
        face_coordinates, eye_coordinates, head_pose_op, gaze_array):
    pre_frame = frame.copy()

    if 'ff' in pre_flag:
        if len(pre_flag) != 1:
            pre_frame = crop_img
        cv2.rectangle(frame, (face_coordinates[0][0], face_coordinates[0][1]), (face_coordinates[0][2], face_coordinates[0][3]),
                      (0, 0, 0), 3)

    if 'fl' in pre_flag:
        cv2.rectangle(crop_img, (eye_coordinates[0][0]-10, eye_coordinates[0][1]-10), (eye_coordinates[0][2]+10, eye_coordinates[0][3]+10),
                      (255, 0, 0), 2)
        cv2.rectangle(crop_img, (eye_coordinates[1][0]-10, eye_coordinates[1][1]-10), (eye_coordinates[1][2]+10, eye_coordinates[1][3]+10),
                      (255, 0, 0), 2)

    if 'fh' in pre_flag:
        cv2.putText(
            frame,
            "Head Pose Angles: yaw= {:.2f} , pitch= {:.2f} , roll= {:.2f}".format(
                head_pose_op[0], head_pose_op[1], head_pose_op[2]),
            (20, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 0, 0), 2)

    if 'fg' in pre_flag:

        cv2.putText(
            frame,
            "Gaze Coordinates: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(
                gaze_array[0], gaze_array[1], gaze_array[2]),
            (20, 80),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 0, 0), 2)

        x, y, w = int(gaze_array[0] * 12), int(gaze_array[1] * 12), 160
        le = cv2.line(left_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
        cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
        re = cv2.line(right_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
        cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
        pre_frame[eye_coordinates[0][1]:eye_coordinates[0][3], eye_coordinates[0][0]:eye_coordinates[0][2]] = le
        pre_frame[eye_coordinates[1][1]:eye_coordinates[1][3], eye_coordinates[1][0]:eye_coordinates[1][2]] = re

    return pre_frame


def main():
    args = build_argparser().parse_args()
    logs = logging.getLogger('main')

    benchmark = False
    # Stored in dictionary for easy access
    path_dictionary = {
        'FaceDetectionModel': args.faceDetectionModel,
        'LandmarkRegressionModel': args.landmarkRegressionModel,
        'HeadPoseEstimationModel': args.headPoseEstimationModel,
        'GazeEstimationModel': args.gazeEstimationModel
    }
    pre_flag = args.previewFlags
    input_file = args.input
    device = args.device
    threshold_prob = args.threshold_prob
    path_output = args.path_output

    if input_file.lower() == 'cam':
        feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_file):
            logs.error("Video file not found\n")
            exit(1)
        feeder = InputFeeder(input_type='video', input_file=input_file)

    for model_path in list(path_dictionary.values()):
        if not os.path.isfile(model_path):
            logs.error("Model file not found\n" + str(model_path))
            exit(1)

    # Load and Instantiate the Models
    face_detection_model = FaceDetection(path_dictionary['FaceDetectionModel'], device, threshold=threshold_prob)
    landmark_detection_model = LandmarkDetectionModel(path_dictionary['LandmarkRegressionModel'], device, threshold=threshold_prob)
    head_pose_estimation_model = HeadPoseEstimationModel(path_dictionary['HeadPoseEstimationModel'], device, threshold=threshold_prob)
    gaze_estimation_model = GazeEstimationModel(path_dictionary['GazeEstimationModel'], device, threshold=threshold_prob)

    if not benchmark:
        mouse_controller = MouseController('medium', 'fast')

    start_load_time = time.time()
    face_detection_model.load_model()
    landmark_detection_model.load_model()
    head_pose_estimation_model.load_model()
    gaze_estimation_model.load_model()
    total_model_load_time = time.time() - start_load_time

    feeder.load_data()

    video_output_file = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), int(feeder.get_fps()/10),
                                (1920, 1080), True)

    frame_count = 0
    inf_time_start = time.time()
    for ret, frame in feeder.next_batch():

        if not ret:
            break

        frame_count += 1

        key = cv2.waitKey(60)

        try:
            face_coordinates, crop_img = face_detection_model.predict(frame)

            if type(crop_img) == int:
                logs.warning("Unable to detect the face")
                if key == 27:
                    break
                continue

            left_eye, right_eye, eye_coordinates = landmark_detection_model.predict(crop_img)
            head_pose_op = head_pose_estimation_model.predict(crop_img)
            mouse_coordinates, gaze_array = gaze_estimation_model.predict(left_eye, right_eye, head_pose_op)

        except Exception as e:
            logs.warning("Prediction failed:" + " " + str(e) + "," + "Frame:" + " " + str(frame_count))
            continue

        image = cv2.resize(frame, (500, 500))

        if not len(pre_flag) == 0:
            pre_frame = draw_preview(
                frame, pre_flag, crop_img, left_eye, right_eye,
                face_coordinates, eye_coordinates, head_pose_op, gaze_array)
            image = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(pre_frame, (500, 500))))

        cv2.imshow('preview', image)
        video_output_file.write(frame)

        if frame_count % 5 == 0 and not benchmark:
            mouse_controller.move(mouse_coordinates[0], mouse_coordinates[1])

        if key == 27:
            break

    time_total = time.time() - inf_time_start
    total_inference_time = round(time_total, 1)
    FPS = frame_count / total_inference_time

    try:
        os.mkdir(path_output)
    except OSError as error:
        logs.error(error)

    with open(path_output+'stats.txt', 'w') as f:
        f.write(str(total_inference_time) + '\n')
        f.write(str(FPS) + '\n')
        f.write(str(total_model_load_time) + '\n')

    logs.info('Model load time: ' + str(total_model_load_time))
    logs.info('Inference time: ' + str(total_inference_time))
    logs.info('FPS: ' + str(FPS))

    logs.info('Video stream ended')
    cv2.destroyAllWindows()
    feeder.close()


if __name__ == '__main__':
    main()
