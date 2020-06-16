"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import os
import cv2
from openvino.inference_engine import IECore


class ModelHeadPoseEstimation:
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

    def predict(self, image):
        """
        This method is meant for running predictions on the input image.
        """
        return self.exec_network.infer(inputs={self.input_blob: image})

    def check_model(self):
        supported_layers = self.plugin.query_network(self.network, self.device)
        for layer in self.network.layers:
            if self.network.layers[layer].name not in supported_layers.keys():
                print(self.network.layers[layer].name + " layer not supported.")
                quit()
        return

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        processed_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_frame = processed_frame.transpose((2, 0, 1))
        processed_frame = processed_frame.reshape(1, *processed_frame.shape)
        return processed_frame

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        return outputs['angle_y_fc'][0][0], outputs['angle_p_fc'][0][0], outputs['angle_r_fc'][0][0]
