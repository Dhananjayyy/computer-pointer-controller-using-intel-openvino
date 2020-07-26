from model import Model


class HeadPoseEstimationModel(Model):
    """
    Head Pose Estimation class
    """

    def __init__(self, path, device='CPU', extensions=None, threshold=0.6):
        """
        Initializes the class
        """
        Model.__init__(self, path, device, extensions, threshold)
        self.model_name = 'Head Pose Estimation'
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def preprocess_output(self, outputs):
        """
        Output:
             "angle_y_fc", shape: [1, 1] - yaw.
             "angle_p_fc", shape: [1, 1] - pitch.
             "angle_r_fc", shape: [1, 1] - roll.
        """
        output_final = []
        try:
            output_final.append(outputs['angle_y_fc'][0][0])
            output_final.append(outputs['angle_p_fc'][0][0])
            output_final.append(outputs['angle_r_fc'][0][0])
        except Exception as e:
            self.logger.error("Pose Estimation Error: Output preprocessing failed" + str(e))
        return output_final

    def predict(self, image, request_id=0):
        """
        Input: Image
        Output: Processed image
        """
        try:
            img = self.preprocess_img(image)
            self.network.start_async(request_id, inputs={self.input_name: img})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs
                finished_output = self.preprocess_output(outputs)
        except Exception as e:
            self.logger.error("Pose Estimation Error: Prediction processing failed" + str(e))
        return finished_output
