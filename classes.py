import tensorflow as tf
import torch
from torchsummary import summary


class Parameters:
    def __init__(self, layer):
        self.layer = layer
        print(f"Parameters initialized for layer: {self.layer.id}")
    def get_params(self):
        return {
            'layer': self.layer.layer
        }        

class Conv2DParams(Parameters):
    def __init__(self, layer, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(layer)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def get_params(self):
        return {
            'layer': self.layer.layer,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding
        }
        
        
class MaxPool2DParams(Parameters):
    def __init__(self, layer, kernel_size=2, stride=2):
        super().__init__(layer)
        self.kernel_size = kernel_size
        self.stride = stride
    
    def get_params(self):
        return {
            'layer': self.layer.layer,
            'kernel_size': self.kernel_size,
            'stride': self.stride
        }
        
class Layer:
    input_id_counter = 0
    conv2d_id_counter = 0
    maxpool2d_id_counter = 0
    linear_id_counter = 0
    dropout_id_counter = 0
    upsample2d_id_counter = 0
    relu_id_counter = 0
    concat_id_counter = 0
    
    def __init__(self, layer_name, params=None):
        
        match layer_name:
            case 'Input':
                self.id = 'x'
                Layer.input_id_counter += 1
            case 'Conv2D':
                self.id = 'Conv2D_'+str(Layer.conv2d_id_counter)
                Layer.conv2d_id_counter += 1
            case 'MaxPool2D':
                self.id = "MaxPool2D_"+str(Layer.maxpool2d_id_counter)
                Layer.maxpool2d_id_counter += 1
            case 'Linear':
                self.id = "Linear_"+ str(Layer.linear_id_counter)
                Layer.linear_id_counter += 1
            case 'Dropout':
                self.id = "Dropout2d_"+ str(Layer.dropout_id_counter)
                Layer.dropout_id_counter += 1
            case 'UpSample2D':
                self.id = "UpSample2D_"+ str(Layer.upsample2d_id_counter)
                Layer.upsample2d_id_counter += 1
            case 'ReLU':
                self.id = "Relu_"+ str(Layer.relu_id_counter)
                Layer.relu_id_counter += 1
            case 'Concat':
                self.id = "Concat_"+ str(Layer.concat_id_counter)
                Layer.concat_id_counter += 1
            case _:
                raise ValueError(f"Unknown layer type: {layer_name}")
        self.layer = self.id.split('_')[0]
        self.params = params
        
    def get_params(self):
        return {
            'id': self.id,
            'layer': self.layer,
            'params': self.params.get_params() if self.params else {}
        }
    