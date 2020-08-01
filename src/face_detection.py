import cv2
import os
import numpy as np
from openvino.inference_engine import IENetwork, IECore

# This class is used for Facial Detection Model

class Model_FaceDetection:

    #Intializing the instance
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name, self.device = model_name, device
        self.core, self.extensions = IECore(), extensions
        self.model_structure = self.model_name
        self.network = self.core.read_network(model=str(model_name),
                                              weights=str(os.path.splitext(model_name)[0] + ".bin"))
        self.model_weights = self.model_name.split('.')[0]+'.bin'
                

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
        s_layers = self.core.query_network(network=self.network, device_name=self.device)
        uns_layers = [layer for layer in self.network.layers.keys() if layer not in s_layers]
        if len(uns_layers) > 0:
            print("Please check extention for these unsupported layers =>" + str(uns_layers))
            exit(1)
        print("Model_FaceDetection layer check")  

    def predict(self, image, prob_threshold):
        '''
        The predtiction method is used to run prediction on images
        '''
        img = image.copy()
        p_img = self.preprocess_input(img)
        d_inf = {self.ip_name:p_img}
        out = self.exec_net.infer(d_inf)
        co_ords = self.preprocess_output(out, prob_threshold)
        l = len(co_ords)
        if (l==0):
            return 0, 0
        co_ords = co_ords[0]
        height, width=image.shape[0], image.shape[1]
        l = [width, height, width, height]
        array = (co_ords* np.array(l))
        co_ords = array.astype(np.int16) 
        a1, b1 = co_ords[1], co_ords[3]
        a2, b2 = co_ords[0], co_ords[2]
        cp_img = image[a1:b1, a2:b2]
        return cp_img, co_ords


    def preprocess_input(self, image):
        '''
        Preprocesses the data before it is fed into inference
        '''
        shape = (self.ip_shape[3], self.ip_shape[2])
        resized = cv2.resize(image, shape)
        x = np.expand_dims(resized,axis=0)
        array = (0,3,1,2)
        p_img = np.transpose(x, array)
        return p_img
            

    def preprocess_output(self, outputs, prob_threshold):
        '''
        Preprocesses the output before feeding it to the model
        '''
        outs = outputs[self.op_name][0][0]
        box =[]
        out = []
        for o in outs:
            conf = o[2]
            if conf>prob_threshold:
                x_min, y_min=o[3], o[4]
                x_max, y_max=o[5], o[6]
                box.append([x_min,y_min,x_max,y_max])
        return box
        

