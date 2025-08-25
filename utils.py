import tensorflow as tf
import torch
from torchsummary import summary
from classes import Layer, Conv2DParams, MaxPool2DParams
import cv2
import numpy as np

class ModelVisualizer:
    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name
        self.model_class_str = f"class {model_name}(torch.nn.Module):\n"+"\tdef __init__(self):\n" + "\t\tsuper().__init__()\n"
        self.forward_str = "\tdef forward(self, x):\n"

    def add_conv2d_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        Adds a Conv2D layer to the model.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Stride of the convolution.
            padding: Padding added to all four sides of the input.
        """
        return f"torch.nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
    
    def add_Linear_layer(self, in_features, out_features):
        """
        Adds a Linear layer to the model.
        
        Args:
            in_features: Number of input features.
            out_features: Number of output features.
        """
        return f"torch.nn.Linear(in_features={in_features}, out_features={out_features})"

    def add_dropout_layer(self, p=0.5):
        """
        Adds a Dropout layer to the model.

        Args:
            p: Probability of an element to be zeroed.
        """
        return f"torch.nn.Dropout(p={p})"
    
    def add_Upsample2d_layer(self, scale_factor=2, mode='nearest'):
        """
        Adds an Upsample layer to the model.
        
        Args:
            scale_factor: The factor by which to upsample the input.
            mode: The algorithm used for upsampling.
        """
        return f"torch.nn.Upsample(scale_factor={scale_factor}, mode='{mode}')"
    
    def add_relu_layer(self, convolutional_layer):
        """
        Adds a ReLU activation layer to the model.
        """
        return f"torch.nn.functional.relu(input={convolutional_layer})"


    def add_maxpool2d_layer(self, convolutional_layer,kernel_size=2, stride=2):
        """
        Adds a MaxPool2d layer to the model.
        
        Args:
            kernel_size: Size of the window to take a max over.
            stride: Stride of the window.
        """
        return f"torch.nn.functional.max_pool2d(input={convolutional_layer}, kernel_size={kernel_size}, stride={stride})"

    def add_concat_layer(self, tensor1, tensor2, dim=1):
        """
        Adds a Concatenate layer to the model.
        
        Args:
            tensor1: First tensor to concatenate.
            tensor2: Second tensor to concatenate.
            dim: Dimension along which to concatenate.
        """
        return f"torch.cat([{tensor1}, {tensor2}], dim={dim})"
    
    def pre_process_input(self, input, out_channel=3, color_space='RGB'):
        if out_channel == "1":
            input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
            return input
        match color_space:
            case 'RGB':
                input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            case 'HSV':
                input = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return input
        
    def generate_model_code(self, input_img, layers):
        print(layers)
        layer_0_params = layers[0]['params']
        img = self.pre_process_input(input_img, out_channel=layer_0_params['out_channels'], color_space=layer_0_params.get('color_space', "RGB"))
        conv2d_layers = []
        relu_layers = []
        maxpool2d_layers = []
        if len(layers) > 2:
            for i, layer in enumerate(layers):
                match layer["layer"].lower():
                    case "conv2d":
                        conv2d_layers.append(layer['id'])
                        params = layer["params"]
                        self.model_class_str += f"\t\tself.{layer['id']} = {self.add_conv2d_layer(params['in_channels'], params['out_channels'], params['kernel_size'], params['stride'], params['padding'])}\n"
                        self.forward_str += f"\t\t{layer['id']} = "+f"self.{layer['id']}"+  f"({params['layer']})\n"
                    case "relu":
                        relu_layers.append(layer['id'])
                        params = layer["params"]
                        self.forward_str += f"\t\t{layer['id']} = {self.add_relu_layer(params['layer'])}\n"
                    case "maxpool2d":
                        maxpool2d_layers.append(layer['id'])
                        params = layer["params"]
                        self.forward_str += f"\t\t{layer['id']} = {self.add_maxpool2d_layer(params['layer'], params['kernel_size'], params['stride'])}\n"
            self.model_class_str += '\n' + self.forward_str
            self.model_class_str += f"\t\treturn {layers[-2]['id']}\n"

            print(self.model_class_str)
            exec(self.model_class_str)
            self.model = eval(f"{self.model_name}()")
            
            with torch.no_grad():
                tensor = torch.from_numpy(img).permute(2, 0, 1) 
                print(summary(self.model, input_size=tensor.shape, batch_size=1, device='cpu'))
                output = self.model(tensor.unsqueeze(0).float())
                img = output.squeeze(0).permute(1, 2, 0).numpy()
                print('img.shape:', img.shape)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return img, self.model_class_str
