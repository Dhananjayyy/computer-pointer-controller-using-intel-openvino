import cv2
import os
import numpy as np
from openvino.inference_engine import IENetwork, IECore

# This class is used for Gaze Estimation Model
class Model_PoseEstimation:
    #Intializing the instance
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name, self.device = model_name, device
        self.core, self.extensions = IECore(), extensions
        self.model_structure = self.model_name
        self.network = self.core.read_network(model=str(model_name),
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
        self.exec_net, self.output_names = load, [i for i in self.network.outputs.keys()]
        self.input_name = next(iter(self.network.input_info))
        self.input_shape = self.network.input_info[self.input_name].input_data.shape
        
    def predict(self, image):
        '''
        The predtiction method is used to run prediction on images
        '''
        img = image.copy()
        proc_img = self.preprocess_input(img)
        d_infer = {self.input_name:proc_img}
        outputs = self.exec_net.infer(d_infer)
        finalOutput = self.preprocess_output(outputs)
        return finalOutput
        

    def check_model(self):
        s_layers = IECore().query_network(network=self.network, device_name=self.device)
        uns_layers = [layer for layer in self.network.layers.keys() if layer not in s_layers]
        if len(uns_layers) > 0:
            print("Please check extention for these unsupported layers =>" + str(uns_layers))
            exit(1)
        print("HeadPoseEstimationModel layer check")

    def preprocess_input(self, image):
        '''
        Preprocesses the data before it is fed into inference
        '''
        shape = (self.input_shape[3], self.input_shape[2])
        image_resized = cv2.resize(image, shape)
        array = (0,3,1,2)
        a_img = np.expand_dims(image_resized,axis=0)
        proc_img = np.transpose(a_img,array)
        return proc_img
            

    def preprocess_output(self, outputs):
        '''
        Preprocesses the out put before feeding it to the model
        '''
        outs = []
        yaw = outputs['angle_y_fc'].tolist()[0][0]
        outs.append(yaw)
        pitch = outputs['angle_p_fc'].tolist()[0][0]
        outs.append(pitch)
        roll = outputs['angle_r_fc'].tolist()[0][0]
        outs.append(roll)
        return outs
