
# Import class from each model's file
from head_pose_estimation import Model_PoseEstimation
from face_detection import Model_FaceDetection
from facial_landmark_detection import Model_LandmarkDetection
from gaze_estimation import Model_GazeEstimation

# Import required modules
import numpy as np
import cv2
import logging
import os
from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
# Set the argument parser
def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-hpm", "--head_pose_model", required=True, type=str,
                        help="Specify Path to Head Pose Estimation model's XML file (include .xml extension)")
    parser.add_argument("-gem", "--gaze_estimation_model", required=True, type=str,
                        help="Specify Path to Gaze Estimation model's XML file (include .xml extension)")
    parser.add_argument("-fdm", "--face_detection_model", required=True, type=str,
                        help="Specify Path to Face Detection model's XML file (include .xml extension)")
    parser.add_argument("-ldm", "--facial_landmark_model", required=True, type=str,
                        help="Specify Path to Facial Landmark Model's XML file (include .xml extension)")
    parser.add_argument("-ip", "--input", required=True, type=str,
                        help="Specify location of video file or enter cam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="Specify CPU extension file locatiom")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify device such as CPU, GPU, FPGA or MYRIAD"
                             "CPU is set by default")
    parser.add_argument("-flags", "--preview_flags", required=False, nargs='+',
                        default=[],
                        help="Provide flags from fdm, ldm, hpm, gem")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="provide the probability threshold to accurately recognize the face")
    return parser


# Main
def main():

    # Get arguments
    args = build_argparser().parse_args()
    cursor = MouseController('medium','fast')
    # Set logger and error messages
    logs = logging.getLogger()
    if args.input.lower()=="cam":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(args.input):
            logs.error("Unable to find specified video file")
        inputFeeder = InputFeeder("video",args.input)
    
    if not os.path.isfile(args.face_detection_model):
      logs.error("Error: face detection model's xml file not found")
    if not os.path.isfile(args.facial_landmark_model):
      logs.error("Error: facial landmark model's xml file not found")
    if not os.path.isfile(args.gaze_estimation_model):
      logs.error("Error: gaze estimation model's xml file not found")
    if not os.path.isfile(args.head_pose_model):
    	logs.error("Error: head pose model's xml file not found")

    # Load and check the model
    Landmark = Model_LandmarkDetection(args.facial_landmark_model, args.device, args.cpu_extension)
    Landmark.check_model()
    fdm = Model_FaceDetection(args.face_detection_model, args.device, args.cpu_extension)
    fdm.check_model()
    hpem = Model_PoseEstimation(args.head_pose_model, args.device, args.cpu_extension)
    hpem.check_model()
    gem = Model_GazeEstimation(args.gaze_estimation_model, args.device, args.cpu_extension)
    gem.check_model()
    load_data = [inputFeeder.load_data(), Landmark.load_model(), fdm.load_model(), hpem.load_model(), gem.load_model()]
    load_data
    
    # Processes
    f_count = 0
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        f_count = f_count+1
        show_video = cv2.imshow('Video',cv2.resize(frame,(500,500)))
        if f_count%5==0: show_video
        prob_thr = args.prob_threshold
        key = cv2.waitKey(60)
        frame_a = frame.copy()
        pred = fdm.predict(frame_a, prob_thr)
        cropped, co_ords = pred
        typ = type(cropped)
        if typ==int:
            logs.error("Face not detected")
            if key==27: break
            continue
        crop_a = cropped.copy()
        pose = hpem.predict(crop_a)
        left_eye, right_eye, box = Landmark.predict(cropped.copy())
        cursor_co, gaze_vector = gem.predict(left_eye, right_eye, pose)
        x, y = box[0][0]-10, box[0][1]-10
        x1, y1 = box[0][2]+10, box[0][3]+10
        x2, y2 = box[1][0]-10, box[1][1]-10
        x3, y3 = box[1][2]+10, box[1][3]+10
        color1 = (255,255,255)
        color2 = (237, 48, 202)
        text_pos, text_pos2 = (10,50), (10,100)
        fontScale = 0.6
        font = 1
        fontColor = (255,255,255)
        lineType = 1
        if (len(args.preview_flags)!=0):
            preview_frame = frame.copy()
            if 'ldm' in args.preview_flags:
                cv2.rectangle(cropped, (x, y), (x1, y1), color1, 2)
                cv2.rectangle(cropped, (x2, y2), (x3, y3), color1, 2)

            if 'fdm' in args.preview_flags:
                cv2.rectangle(preview_frame, (co_ords[0], co_ords[1]), (co_ords[2], co_ords[3]), (255,0,0), 3)
                preview_frame = cropped
            if 'gem' in args.preview_flags:
                x = int(gaze_vector[0]*12)
                y = int(gaze_vector[1]*12)
                w = 160
                le = left_eye.copy()
                re = right_eye.copy()
                thick = 2
                start_a, end_a = (x-w, y-w), (x+w, y+w)
                start_b, end_b = (x-w, y+w), (x+w, y-w)
                cv2.line(left_eye, start_b, end_b, color2, thick)
                cv2.line(right_eye, start_b, end_b, color2, thick)
                left = cv2.line(le, start_a, end_a, color2, thick)
                right = cv2.line(re, start_a, end_a, color2, thick)
                cv2.line(re, start_a, end_a, color2, thick)
                a1, b1, c1, d1 = box[0][0], box[0][1], box[0][2], box[0][3]
                a2, b2, c2, d2 = box[1][0], box[1][1], box[1][2], box[1][3]
                cropped[b1:d1,a1:c1], cropped[b2:d2,a2:c2] = left, right
        
            if 'hpm' in args.preview_flags:
                cv2.putText(preview_frame, """Angles: Roll= {:.1f} , Pitch= {:.1f} , Yaw= {:.1f} """.format(pose[2], pose[1], pose[0]),
            	text_pos,
            	font,
            	fontScale, fontColor, lineType)
            
            show_video = cv2.imshow("",cv2.resize(preview_frame,(500,500)))
            show_video
        if f_count%5==0: cursor.move(cursor_co[0],cursor_co[1])    
        if key==27: break
    # Ends program
    end_msg, end_feed, end_video = logs.error("Video ended."), inputFeeder.close(), cv2.destroyAllWindows()
    end_msg
    end_feed
    end_video
if __name__ == '__main__':
    main() 