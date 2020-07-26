from model import Model
import math


class GazeEstimationModel(Model):
    """
    Gaze Estimation Class
    """

    def __init__(self, path, device='CPU', extensions=None, threshold=0.6):
        """
        Initializes the class
        """
        Model.__init__(self, path, device, extensions, threshold)
        self.model_name = 'Face Detection'
        self.input_name = [i for i in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        self.output_name = [i for i in self.model.outputs.keys()]

    def preprocess_output(self, outputs, head_pose_coords):
        """
        Output: Dictionary like {'gaze_array': array([[ 0.54552154,  0.41245478, -0.21244471]], dtype=float32)}
        Converts: head_pose_coords from Radiam to Catesian coordinates.
        """
        gaze_array = outputs[self.output_name[0]][0]
        mouse_coordinates = (0, 0)
        try:
            angle = head_pose_coords[2]
            X = gaze_array[0] * (math.cos(angle * math.pi / 180.0)) + gaze_array[1] * (math.sin(angle * math.pi / 180.0))
            Y = -gaze_array[0] * (math.sin(angle * math.pi / 180.0)) + gaze_array[1] * (math.cos(angle * math.pi / 180.0))
            mouse_coordinates = (X, Y)
        except Exception as e:
            self.logger.error("Gaze Estimation Error: Output preprocessing failed:\n" + str(e))
        return mouse_coordinates, gaze_array

    def predict(self, left_eye, right_eye, head_pose_coords, request_id=0):
        """
        Input: Image
        Output: Processed Image
        """
        try:
            left_eye = self.preprocess_img(left_eye)
            right_eye = self.preprocess_img(right_eye)
            self.network.start_async(request_id, inputs={'left_eye_image': left_eye,
                                                         'right_eye_image': right_eye,
                                                         'head_pose_angles': head_pose_coords})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs
                mouse_coordinates, gaze_array = self.preprocess_output(outputs, head_pose_coords)
        except Exception as e:
            self.logger.error("Gaze Estimation Error: Prediction processing failed:\n" + str(e))
        return mouse_coordinates, gaze_array

    
