from model import Model


class LandmarkDetectionModel(Model):
    """
    Landmark Detection Class
    """

    def __init__(self, path, device='CPU', extensions=None, threshold=0.6):
        """
        Initializes the class
        """
        Model.__init__(self, path, device, extensions, threshold)
        self.model_name = 'Landmark Detection'
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))

    def preprocess_output(self, outputs, image):
        """
        Output: blob with the shape: [1, 10],
        Contains an array of 10 float values,
        The coordinates are normalized.
        """
        h = image.shape[0]
        w = image.shape[1]
        left_eye, right_eye, eye_coordinates = [], [], []
        try:
            outputs = outputs[0]

            left_xmin = int(outputs[0][0][0] * w) - 10
            left_ymin = int(outputs[1][0][0] * h) - 10
            right_xmin = int(outputs[2][0][0] * w) - 10
            right_ymin = int(outputs[3][0][0] * h) - 10

            left_xmax = int(outputs[0][0][0] * w) + 10
            left_ymax = int(outputs[1][0][0] * h) + 10
            right_xmax = int(outputs[2][0][0] * w) + 10
            right_ymax = int(outputs[3][0][0] * h) + 10

            left_eye = image[left_ymin:left_ymax, left_xmin:left_xmax]
            right_eye = image[right_ymin:right_ymax, right_xmin:right_xmax]

            eye_coordinates = [[left_xmin, left_ymin, left_xmax, left_ymax],
                         [right_xmin, right_ymin, right_xmax, right_ymax]]

        except Exception as e:
            self.logger.error("Landmark Detection Error: Could not draw bounding boxes:\n" + str(e))
        return left_eye, right_eye, eye_coordinates

    def predict(self, image, request_id=0):
        """
        Input: Image
        Output: Processed Image
        """
        left_eye, right_eye, eye_coordinates = [], [], []
        try:
            img = self.preprocess_img(image)
            self.network.start_async(request_id, inputs={self.input_name: img})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs[self.output_name]
                left_eye, right_eye, eye_coordinates = self.preprocess_output(outputs, image)
        except Exception as e:
            self.logger.error("Landmark Detection Error: Predictions failed:\n" + str(e))
        return left_eye, right_eye, eye_coordinates

    