"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import os
import cv2
from openvino.inference_engine import IECore


class ModelGazeEstimation:
    """
    Class for the Face Detection Model.
    """
    def __init__(self, model_name, device='CPU', extensions=None, probability_threshold=0.5):
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.model = model_name
        self.device = device
        self.extensions = extensions
        self.probability_threshold = probability_threshold

    def load_model(self):
        """
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where it can load them.
        """
        model_xml = self.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.plugin = IECore()
        if self.extensions and "CPU" in self.device:
            self.plugin.add_extension(self.extensions, self.device)
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)
        self.check_model()
        self.exec_network = self.plugin.load_network(self.network, self.device)
        self.input_blob = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.network.outputs))
        return

    def predict(self, input_object):
        """
        This method is meant for running predictions on the input image.
        """
        return self.exec_network.infer(inputs=input_object)

    def check_model(self):
        supported_layers = self.plugin.query_network(self.network, self.device)
        for layer in self.network.layers:
            if self.network.layers[layer].name not in supported_layers.keys():
                print(self.network.layers[layer].name + " layer not supported.")
                quit()
        return

    def preprocess_input(self, face_image, face_landmarks, head_pose_angles):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        cropped_face_width = face_image.shape[1]
        cropped_face_height = face_image.shape[0]

        right_eye_landmark = (int(face_landmarks[0] * cropped_face_width), int(face_landmarks[1] * cropped_face_height))
        left_eye_landmark = (int(face_landmarks[2] * cropped_face_width), int(face_landmarks[3] * cropped_face_height))

        cropping_radius = int((left_eye_landmark[0]-right_eye_landmark[0])/2)
        cropped_left_eye = face_image[
                           max(left_eye_landmark[1]-cropping_radius, 0):min(left_eye_landmark[1]+cropping_radius, cropped_face_width),
                           max(left_eye_landmark[0]-cropping_radius, 0):min(left_eye_landmark[0]+cropping_radius, cropped_face_height)
                           ]
        cropped_right_eye = face_image[
                           max(right_eye_landmark[1]-cropping_radius, 0):min(right_eye_landmark[1]+cropping_radius, cropped_face_width),
                           max(right_eye_landmark[0]-cropping_radius, 0):min(right_eye_landmark[0]+cropping_radius, cropped_face_height)
                           ]

        processed_left_eye = cv2.resize(cropped_left_eye, (self.network.inputs['left_eye_image'].shape[3], self.network.inputs['left_eye_image'].shape[2]))
        processed_left_eye = processed_left_eye.transpose((2, 0, 1))
        processed_left_eye = processed_left_eye.reshape(1, *processed_left_eye.shape)
        processed_right_eye = cv2.resize(cropped_right_eye, (self.network.inputs['right_eye_image'].shape[3], self.network.inputs['right_eye_image'].shape[2]))
        processed_right_eye = processed_right_eye.transpose((2, 0, 1))
        processed_right_eye = processed_right_eye.reshape(1, *processed_right_eye.shape)

        return {
            'left_eye_image': processed_left_eye,
            'right_eye_image': processed_right_eye,
            'head_pose_angles': head_pose_angles
        }

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        return outputs[self.output_blob][0]
