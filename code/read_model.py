from collections import namedtuple
import os, sys
import torch

def read_fc_model(net:nn.Module):
    """
    read fully connected model
    
    args:
    net - torch loaded model

    returns:
    parameters of all layers of the model  
    """
    # named tuple for the linear layers
    linear = namedtuple("linear",['id','w', 'b'])
    
    layers = net.layers

    mean, var = layers[0].mean.item(), layers[0].var.item()

    output = []
    norm_layer = linear(0, 'w'=1/var, 'b'= -mean/var)
    output.append(norm_layer)

    for i, l in enumerate(layers):
        if isinstance(l, nn.Linear):
            ll = linear(i, 'w':l.weights, 'b':l.bias)
            output.append(ll)
    
    return output

def read_conv_model(net:nn.Module):
    """
    read convolution model
    
    args:
    net - torch loaded model

    returns:
    parameters of all layers of the model  
    """
    pass




