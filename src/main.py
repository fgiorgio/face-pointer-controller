import cv2
import math
import numpy as np
import logging
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import ModelFaceDetection
from facial_landmarks_detections import ModelFacialLandmarksDetections
from gaze_estimation import ModelGazeEstimation
from head_pose_estimation import ModelHeadPoseEstimation
from mouse_controller import MouseController


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
    parser.add_argument("-mp", "--mouse_precision", type=str, default='medium',
                        help="Mouse controller precision: low, medium (default), high")
    parser.add_argument("-ms", "--mouse_speed", type=str, default='medium',
                        help="Mouse controller speed: slow, medium (default), fast")
    return parser


def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix


def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1, 0], [math.sin(yaw), 0, math.cos(yaw)]])
    rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    # R = np.dot(Rz, Ry, Rx)
    # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    # R = np.dot(Rz, np.dot(Ry, Rx))
    r = rz @ ry @ rx
    # print(R)
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(r, xaxis) + o
    yaxis = np.dot(r, yaxis) + o
    zaxis = np.dot(r, zaxis) + o
    zaxis1 = np.dot(r, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (255, 0, 0), 2)
    # cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_argparser().parse_args()

    input_feeder = InputFeeder(args.input_type, args.input_path)
    input_feeder.load_data()
    logging.info('Input ready')

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
    logging.info('Models loaded')

    logging.info('Starting inference')
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

        cropped_face = frame[y_min:y_max, x_min:x_max].copy()
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

        # HEAD POSE ESTIMATOR
        head_pose_estimator_frame = head_pose_estimator.preprocess_input(cropped_face)
        head_pose_estimator_output = head_pose_estimator.predict(head_pose_estimator_frame)
        head_pose_estimator_prediction = head_pose_estimator.preprocess_output(head_pose_estimator_output)
        if args.intermediate_output is not None:
            yaw, pitch, roll = head_pose_estimator_prediction
            focal_length = 950.0
            scale = 100
            center_of_face = (
                x_min + int(face_detector_prediction[4] * cropped_face_width),
                y_min + int(face_detector_prediction[5] * cropped_face_height),
                0
            )
            draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length)

        # GAZE ESTIMATOR
        gaze_estimator_input = gaze_estimator.preprocess_input(
            cropped_face,
            face_detector_prediction,
            head_pose_estimator_prediction
        )
        gaze_estimator_output = gaze_estimator.predict(gaze_estimator_input)
        gaze_estimator_prediction = gaze_estimator.preprocess_output(gaze_estimator_output)
        logging.info('Output: ' + str(gaze_estimator_prediction[:2]))

        # MOUSE CONTROLLER
        mouse_controller = MouseController(args.mouse_precision, args.mouse_speed)
        mouse_controller.move(gaze_estimator_prediction[0], gaze_estimator_prediction[1])

        cv2.imshow('frame', frame)

        if key_pressed & 0xFF == ord('q'):
            logging.info('Inference stopped')
            break

    input_feeder.close()
    logging.info('Inference finished')


if __name__ == '__main__':
    main()
