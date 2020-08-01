import cv2
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import os

# This class is used for Facial Landmarks Detection Model

class Model_LandmarkDetection:

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
        
        uns_layers = []
        for x in self.network.layers.keys():
            if x not in s_layers:
                uns_layers = x

        l = len(uns_layers)
        if l !=0 and self.device=='CPU':
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
        self.exec_net = load
        '''
        Oputput name and Output shape
        '''
        o_name = next(iter(self.network.outputs))
        self.op_name = o_name
        o_shape = self.network.outputs[self.op_name].shape
        self.op_shape = o_shape
        '''
        Input name and Input shape 
        '''
        i_name = next(iter(self.network.input_info))
        self.ip_name = i_name
        i_shape = self.network.input_info[self.ip_name].input_data.shape
        self.ip_shape = i_shape
        

    def check_model(self):
        s_layers = IECore().query_network(network=self.network, device_name=self.device)
        uns_layers = [layer for layer in self.network.layers.keys() if layer not in s_layers]
        if len(uns_layers) > 0:
            print("Please check extention for these unsupported layers =>" + str(uns_layers))
            exit(1)
        print("FacialLandmarksDetectionModel layer check")

    def predict(self, image):
        '''
        The predtiction method is used to run prediction on images
        '''
        img = image.copy()
        p_img = self.preprocess_input(img)
        d_inf = {self.ip_name:p_img}
        out = self.exec_net.infer(d_inf) 
        height, width=image.shape[0], image.shape[1]
        coords = ((self.preprocess_output(out))* np.array([width, height, width, height])).astype(np.int32)
        # Getting left eye co-ordinates
        left_x_min, left_y_min = coords[0]-10, coords[1]-10
        left_x_max, left_y_max = coords[0]+10, coords[1]+10
        # Getting right eye co-ordinates
        right_x_min, right_y_min=coords[2]-10, coords[3]-10
        right_x_max, right_y_max=coords[2]+10, coords[3]+10
        # Getting left and right eye co-ordinates
        l_co_ords, r_co_ords =  image[left_y_min:left_y_max, left_x_min:left_x_max], image[right_y_min:right_y_max, right_x_min:right_x_max]
        # def get_coords(self,image)
        # Getting total co-ordinates
        a, b = [left_x_min,left_y_min,left_x_max,left_y_max], [right_x_min,right_y_min,right_x_max,right_y_max]
        co_ords= [a,b]
        total = co_ords
        return l_co_ords, r_co_ords, total


    def preprocess_input(self, image):
        '''
        Preprocesses the data before it is fed into inference
        '''
        conv = cv2.COLOR_BGR2RGB
        col = cv2.cvtColor(image, conv)
        shape = (self.ip_shape[3], self.ip_shape[2])
        res = cv2.resize(col, shape)
        x = np.expand_dims(res,axis=0)
        array = (0,3,1,2)
        p_img = np.transpose(x, array)
        return p_img
            

    def preprocess_output(self, outputs):
        '''
        Preprocesses the out put before feeding it to the model
        '''
        outs = outputs[self.op_name][0]
        left_x, left_y = outs[0].tolist()[0][0], outs[1].tolist()[0][0]
        right_x, right_y = outs[2].tolist()[0][0], outs[3].tolist()[0][0]
        output = (left_x, left_y, right_x, right_y)
        return output
