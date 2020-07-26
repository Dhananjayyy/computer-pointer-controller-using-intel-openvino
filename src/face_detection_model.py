from model import Model
import numpy as np

class FaceDetection(Model):
    """
    Face Detection Class
    """

    def __init__(self, path, device='CPU', extensions=None, threshold=0.8):
        """
        Initializes the class
        """
        Model.__init__(self, path, device, extensions, threshold)
        self.model_name = 'Face Detection'
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def preprocess_output(self, coords, image):
        """
        Input: Image and preprocessed coordinates
        Output: Bounding boxed images with cordinates
        """
        detection = []
        crop_img = image
        w = int(image.shape[1])
        h = int(image.shape[0])
        coords = np.squeeze(coords)
        try:
            for coordinate in coords:
                image_id, label, threshold, xmin, ymin, xmax, ymax = coordinate
                if image_id == -1:
                    break
                if label == 1 and threshold >= self.threshold:
                    xmin = int(xmin * w)
                    ymin = int(ymin * h)
                    xmax = int(xmax * w)
                    ymax = int(ymax * h)
                    detection.append([xmin, ymin, xmax, ymax])
                    crop_img = image[ymin:ymax, xmin:xmax]
        except Exception as e:
            self.logger.error("Face Detection Error: Could not draw bounding boxes:\n" + str(e))
        return detection, crop_img


    def predict(self, image, request_id=0):
        """
        Input: Image
        Output: Image Detection,Crop Image
        """
        try:
            img = self.preprocess_img(image)
            self.network.start_async(request_id, inputs={self.input_name: img})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs[self.output_name]
                detection, crop_img = self.preprocess_output(outputs, image)
        except Exception as e:
            self.logger.error("Face Detection Error : Predictions failed:\n" + str(e))
        return detection, crop_img
