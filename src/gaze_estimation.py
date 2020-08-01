import cv2
import numpy as np
import os
from openvino.inference_engine import IENetwork, IECore
import math

# This class is used for Gaze Estimation Model
class Model_GazeEstimation:
    #Intializing the instance
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name, self.device = model_name, device
        self.core, self.extensions = IECore(), extensions
        self.model_structure = self.model_name
        self.network = IECore().read_network(model=str(model_name),
                                              weights=str(os.path.splitext(model_name)[0] + ".bin"))
        self.model_weights = self.model_name.split(".")[0]+'.bin'
        

    def load_model(self):
        '''
        load_model mothod is for loading the model to the device.
        '''
        # Initializes the network
        # Supported Layers
        s_layers = self.core.query_network(network=self.network, device_name=self.device)
        # Unsupported Layers
        uns_layers = []
        for x in self.network.layers.keys():
            if x not in s_layers:
                uns_layers = x
        
        l = len(uns_layers)
        if l!=0 and self.device=='CPU':
            print("Layer "+uns_layers+" not supported")

            if not self.extensions==None:
                print("Add cpu extension layer")
                # Adding extension
                add_ext = self.core.add_extension(self.extensions, self.device)
                add_ext
                # Supported and Unsupported layers
                s_layers = self.core.query_network(network = self.network, device_name=self.device)
                uns_layers = []
                for x in self.network.layers.keys():
                    if x not in s_layers:
                        uns_layers = x
                if not l==0:
                    print("Layer not supported")
                    exit(1)
            else:
                print("Specify path of cpu extension")
                exit(1)
        # Intializing network
        load = self.core.load_network(network=self.network, device_name=self.device,num_requests=1)
        self.exec_net, self.input_name = load, [i for i in self.network.input_info.keys()]
        self.input_shape, self.output_names = self.network.input_info[self.input_name[1]].input_data.shape, [i for i in self.network.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        The predtiction method is used to run prediction on images
        '''
        l_img = left_eye_image.copy()
        r_img = right_eye_image.copy()
        proc_l_img, proc_r_img = self.preprocess_input(l_img, r_img)
        inf = {'head_pose_angles':hpa, 'left_eye_image':proc_l_img, 'right_eye_image':proc_r_img}
        outputs = self.exec_net.infer(inf)
        cursor, array = self.preprocess_output(outputs,hpa)
        return cursor, array

    def check_model(self):
        s_layers = IECore().query_network(network=self.network, device_name=self.device)
        uns_layers = [layer for layer in self.network.layers.keys() if layer not in s_layers]
        if len(uns_layers) > 0:
            print("Please check extention for these unsupported layers =>" + str(uns_layers))
            exit(1)
        print("GazeEstimation layer Check")

    def preprocess_input(self, left_eye, right_eye):
        '''
        Preprocesses the data before it is fed into inference
        '''
        shape = (self.input_shape[3], self.input_shape[2])
        l_resized = cv2.resize(left_eye, shape)
        r_resized = cv2.resize(right_eye, shape)
        array = (0,3,1,2)
        a = np.expand_dims(l_resized,axis=0)
        b = np.expand_dims(r_resized,axis=0)
        proc_l_img = np.transpose(a, array)
        proc_r_img = np.transpose(b, array)
        return proc_l_img, proc_r_img
            

    def preprocess_output(self, outputs,hpa):
        '''
        Preprocesses the output before feeding it to the model
        '''
        array = outputs[self.output_names[0]].tolist()[0]
        conv = math.pi / 180.0
        cos = math.cos(hpa[2] * conv)
        sin = math.sin(hpa[2] * conv)
        a1, b1 = array[0] * cos, array[1] * sin
        a2, b2 = array[0] * sin, array[1] * cos
        x = a1 + b1
        y = -a2 + b2
        return (x,y), array
        
        
