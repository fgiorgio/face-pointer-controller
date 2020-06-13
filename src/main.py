import os
import sys
import time
import socket
import json
import cv2
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import ModelFaceDetection
from facial_landmarks_detections import ModelFacialLandmarksDetections
from gaze_estimation import ModelGazeEstimation
from head_pose_estimation import ModelHeadPoseEstimation


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-t", "--input_type", type=str, default="cam",
                        help="Input type: cam (default), image, video")
    parser.add_argument("-p", "--input_path", type=str, default=None,
                        help="Absolute path to image or video file")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: CPU (default), GPU, FPGA, MYRIAD")
    parser.add_argument("-e", "--cpu_extension", type=str, default=None,
                        help="Absolute path to a shared library with the kernels impl.")
    parser.add_argument("-mf", "--model_face_detector", type=str, default=None,
                        help="Absolute path to the Face Detector model xml file")
    parser.add_argument("-mfl", "--model_facial_landmarks_detector", type=str, default=None,
                        help="Absolute path to the Facial Landmarks Detector model xml file")
    parser.add_argument("-mg", "--model_gaze_estimator", type=str, default=None,
                        help="Absolute path to the Gaze Estimator model xml file")
    parser.add_argument("-mhp", "--model_head_pose_estimator", type=str, default=None,
                        help="Absolute path to the Head Pose Estimator model xml file")
    parser.add_argument("-pf", "--prob_face_detector", type=float, default=0.5,
                        help="Probability threshold for Face Detector model detections (0.5 by default)")
    parser.add_argument("-pfl", "--prob_facial_landmarks_detector", type=float, default=0.5,
                        help="Probability threshold for Facial Landmarks Detector model detections (0.5 by default)")
    parser.add_argument("-pg", "--prob_gaze_estimator", type=float, default=0.5,
                        help="Probability threshold for Gaze Estimator model estimations (0.5 by default)")
    parser.add_argument("-php", "--prob_head_pose_estimator", type=float, default=0.5,
                        help="Probability threshold for Head Pose Estimator model estimations (0.5 by default)")
    parser.add_argument("-io", "--intermediate_output", nargs='?', const='',
                        help="Show the outputs of intermediate models")
    return parser


def main():
    args = build_argparser().parse_args()

    input_feeder = InputFeeder(args.input_type, args.input_path)
    input_feeder.load_data()

    face_detector = ModelFaceDetection(
        args.model_face_detector,
        args.device,
        args.cpu_extension,
        args.prob_face_detector
    )
    facial_landmarks_detector = ModelFacialLandmarksDetections(
        args.model_facial_landmarks_detector,
        args.device,
        args.cpu_extension,
        args.prob_facial_landmarks_detector
    )
    gaze_estimator = ModelGazeEstimation(
        args.model_gaze_estimator,
        args.device,
        args.cpu_extension,
        args.prob_gaze_estimator
    )
    head_pose_estimator = ModelHeadPoseEstimation(
        args.model_head_pose_estimator,
        args.device,
        args.cpu_extension,
        args.prob_head_pose_estimator
    )

    face_detector.load_model()
    facial_landmarks_detector.load_model()
    gaze_estimator.load_model()
    head_pose_estimator.load_model()

    for frame in input_feeder.next_batch():
        key_pressed = cv2.waitKey(1)

        if frame is None:
            break

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # FACE DETECTOR
        face_detector_frame = face_detector.preprocess_input(frame)
        face_detector_output = face_detector.predict(face_detector_frame)
        face_detector_prediction = face_detector.preprocess_output(face_detector_output)
        x_min = int(face_detector_prediction[3] * frame_width)
        y_min = int(face_detector_prediction[4] * frame_height)
        x_max = int(face_detector_prediction[5] * frame_width)
        y_max = int(face_detector_prediction[6] * frame_height)
        if args.intermediate_output is not None:
            frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

        cropped_face = frame[y_min:y_max, x_min:x_max]
        cropped_face_width = cropped_face.shape[1]
        cropped_face_height = cropped_face.shape[0]

        # FACIAL LANDMARKS DETECTOR
        facial_landmarks_detector_frame = facial_landmarks_detector.preprocess_input(cropped_face)
        facial_landmarks_detector_output = facial_landmarks_detector.predict(facial_landmarks_detector_frame)
        face_detector_prediction = facial_landmarks_detector.preprocess_output(facial_landmarks_detector_output)
        if args.intermediate_output is not None:
            for i in range(0, 10, 2):
                x = x_min + int(face_detector_prediction[i] * cropped_face_width)
                y = y_min + int(face_detector_prediction[i+1] * cropped_face_height)
                frame = cv2.circle(frame, (x, y), 2, (255, 255, 255), 2)

        cv2.imshow('frame', frame)

        if key_pressed & 0xFF == ord('q'):
            break

    input_feeder.close()


if __name__ == '__main__':
    main()
