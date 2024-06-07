from nf_model.network import *
from nf_model.layer import *

def dnn(input_shape=(None, 13), nb_classes=2,get_weights = False):
    """
    The implementation of a DNN model
    :param input_shape: the shape of dataset
    :param nb_classes: the number of classes
    :return: a DNN model
    """
    activation = ReLU
    linear64, linear32, linear16, linear8, linear4, linear_classify = Linear(64), Linear(32), Linear(16), Linear(
        8), Linear(4), Linear(nb_classes)
    layers = [linear64,
              activation(),
              linear32,
              activation(),
              linear16,
              activation(),
              linear8,
              activation(),
              linear4,
              activation(),
              linear_classify,
              Softmax()]
       
    model = MLP(layers, input_shape)
    if get_weights:
        return model, [linear64.W, linear32.W, linear16.W, linear8.W, linear4.W, linear_classify.W]
    else:
        return model