import logging
import cv2
from openvino.inference_engine import IECore, IENetwork


class Model:
    def __init__(self, path, device='CPU', extensions=None, threshold=0.7):
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None
        self.network = None
        self.model_structure = path
        self.model_weights = path.replace('.xml', '.bin')
        self.device_name = device
        self.threshold = threshold
        self.logger = logging.getLogger('fd')
        self.model_name = 'Model'
        try:
            self.core = IECore()
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            self.logger.error("Error: Initilization failed for" + " " + str(self.model_name) + str(e))
            raise ValueError("Error: Failed to initialise the network. Please enter the correct model path.")

    def load_model(self):
        """
        Returns: Loaded model (IECore) 
        """
        try:
            self.network = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
        except Exception as e:
            self.logger.error("Error: Loading failed for"+ " " + str(self.model_name)+str(e))

    def predict(self):
        pass

    def preprocess_output(self):
        pass

    def preprocess_img(self, image):
        """
        Input: image
        Steps:
            1. Resizing according to the input shape
            2. Transpose of image
            3. Reshaping of image
        Output: Preprocessed image
        """
        try:
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)
        except Exception as e:
            self.logger.error("Error: Preprocessing image failed for" + " " + str(self.model_name) + str(e))
        return image

    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.network.requests[0].wait(-1)
        return status