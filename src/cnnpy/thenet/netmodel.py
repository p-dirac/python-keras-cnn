import os
from pathlib import Path
import keras
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Input
from keras import Sequential
from keras.optimizers import SGD
from keras.models import load_model
#
from PySide6.QtCore import Signal, QObject
#
from appctrl.app_util import Util

import json
#
import logging
logger = logging.getLogger()

class NetModel(QObject):
    """
    Model for convolution network.

    """
    # Class variables
    # send signal when params has been set
    updateStatus = Signal(str)
    loadComplete = Signal()
    #
    # app results folder, which stores result files 
    APP_RESULTS_SUBDIR = "Results/app_results"

    def __init__(self):
        # must include super call for QObject, Signal
        super().__init__()


    def setInfo(self, in_shape: tuple, num_classes: int, hyper_params: dict):
        """Set various factors for specifying a neural network model

        Args:
            in_shape (tuple): 3D tensor (height, width, channels), which specifies the shape of a single sample
            num_classes (int): number of classes to be classified
            hyper_params (dict): hyper parameters required to define the model
        """
        #
        # model: keras training model
        self.model = None
        self.num_classes = num_classes
        self.in_shape = in_shape
        self.hyper = hyper_params
        #print("setInfo, hyper: ", self.hyper)

    def getModel(self):
        return self.model

    def createModel(self):
        """Create new neural network model, which will be trained when model fit is called

        Note 1: setInfo must be called before createModel
        Note 2: a subset of the hyper params are in list format for tuning, and here only the first item is used with a [0] suffix. For example, hyper["num_filters"][0].

        Returns:
            Sequential : neural network model
        """
        try:
            # clear out old models and layers from the global state
            keras.backend.clear_session()
            # define new model
            self.model = Sequential()
            # input format (height, width, Number Channels)
            print("createModel, in_shape: ", self.in_shape)
            # Model needs to know the shape of input into the first layer.
            # The following layers can do automatic shape inference.
            self.model.add(Input(shape=self.in_shape))
            #
            # convolution: to emphasize image patterns 
            # small matrix slides over an input matrix
            # multiply and sum to produce output matrix
            # 1st convolution matrix size 3 by 3 or 5x5
            self.model.add(
                Conv2D(filters=self.hyper["num_filters"][0], kernel_size=self.hyper["kernel_size"], strides=1, padding="Same", activation="relu", kernel_initializer="he_uniform")
            )
            #
            # 1st pooling layer
            # selects max value from each set of overlapping filters
            # to reduce size of ouput matrix
            self.model.add(MaxPool2D((2, 2)))
            #
            # 2nd convolution matrix size 3 by 3 or 5x5
            self.model.add(
                Conv2D(filters=self.hyper["num_filters"][0], kernel_size=self.hyper["kernel_size"], strides=1, padding="Same", activation="relu", kernel_initializer="he_uniform")
            )
            # 2nd pooling layer
            self.model.add(MaxPool2D((2, 2)))
            #
            # reshapes n-dimensional input tensor into a one-dimensional tensor
            self.model.add(Flatten())
            #
            # dense layer connects every input feature to every neuron in that layer
            # every neuron computes a weighted sum of all inputs, applies an activation function, and passes result to the next layer
            self.model.add(Dense(units=self.hyper["num_units"][0], activation="relu", kernel_initializer="he_uniform"))
            #
            # dropout layer to increase accuracy
            # overfitting is reduced by randomly omitting half of the neurons  from the previous layer before passing to the next layer
            self.model.add(Dropout(self.hyper["drop_out"]))
            #
            # output layer
            # softmax normalizes output scores into probabilities that sum to one
            self.model.add(Dense(units=self.num_classes, activation="softmax"))
            #
            # learning rate schedule
            # allows larger weight updates during the beginning of training
            # and smaller updates later in the training
            # Note: decay_steps is number of batches processed
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.hyper["init_rate"][0],
                decay_steps=self.hyper["decay_steps"],
                decay_rate=self.hyper["decay_rate"]
            )
            #
            # define optimizer and loss 
            # calculate loss function 
            # pk = predicted probability for correct class k
            # sparse_categorical_crossentropy loss: 
            #   -(1/N)sumOverN(log pki) for sum i = 1 to N, over batch size N
            #
            opt = SGD(learning_rate=lr_schedule, momentum=self.hyper["momentum"])
            self.model.compile(
                optimizer=opt,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        except Exception:
            logger.error("Error in createModel")
            raise Exception("Error in createModel")
        else:
            # model type is keras Sequential
            return self.model

    def loadModel(self):
        """Read cnn model from .keras file, and save in class variable
        
        The model .keras file stores the architecture and weights in
        a single file.

        Args:
            None

        Raises:
            Exception: if error occurs

        """
        try:
            cwd = Path.cwd()
            # cwd.parents[1] for 2 levels up from current directory
            full_dir = os.path.join(cwd.parents[1], NetModel.APP_RESULTS_SUBDIR)
            file_path = Util.openFileDialog(title="Open .keras model file", filters=['*.keras'], dir=full_dir)
            # Read from file and parse h5
            print("loadModel, file_path: ", file_path)
            # Note: file_path will be None if user cancels file dialog
            if(file_path is not None):
                if os.path.exists(str(file_path)):
                    # open file for reading
                    #with open(file_path, "r") as f:
                    # read model from .keras file 
                    self.model = load_model(str(file_path))
                    self.updateStatus.emit("CNN Model loaded") 
                    self.loadComplete.emit()    
                else:
                    # file does not exist
                    Util.alert(f"Model .keras file not found: {str(file_path)}")     
        except Exception:
            logger.error("Error in loadModel")
            raise Exception("Error in loadModel") 

    def saveModel(self):
        """Saves neural network model in keras file format
        """        
        if (self.model is None) or (not self.model) :
            Util.alert("CNN model is not set.")
        else:    
            # save model 
            self.saveModelToKeras()
            # send signal to update status
            self.updateStatus.emit("CNN Model saved")

    def saveModelToKeras(self):
        """Write CNN model to Keras file with .keras extension, which stores the
        architecture and weights in a single file.

        Raises:
            Exception: if error occurs

        """
        try:
            cwd = Path.cwd()
            # cwd.parents[1] for 2 levels up from current directory
            full_dir = os.path.join(cwd.parents[1], NetModel.APP_RESULTS_SUBDIR)
            file_path = Util.openFileDialog(title="Save As .keras:", filters=['*.keras'], dir=full_dir)
            print("saveModelToH5, file_path: ", file_path)
            if file_path:
                # open file for writing
                #with open(file_path, "w") as f:
                    #write model to file in h5 format
                    # e.g. cnn_model.keras
                self.model.save(str(file_path))
            else:
                # user canceled dialog without selecting a path
                logger.error("No path selected, model not saved")
                        
        except Exception:
            logger.error("Error in saveModelToKeras")
            raise Exception("Error in saveModelToKeras")
            Util.alert("Error in saveModelToKeras.")

    def readHistory(self, history_file: str):
        """Read history dictionary from file

        Args:
            history_file (str): history dictionary file name

        Raises:
            Exception: if error occurs

        Returns:
            dict : history dictionary
        """
        try:
            cwd = Path.cwd()
            # cwd.parents[1] for 2 levels up from current directory
            full_dir = os.path.join(cwd.parents[1], NetModel.APP_RESULTS_SUBDIR)
            file_path = Util.openFileDialog(title="Open JSON history file", filters=['*.json'], dir=full_dir)
            # Read from file and parse JSON
            # open file for reading
            with file_path.open("r") as f:
                # deserialize the hist_dict from file f 
                hist_dict = json.load(f)
        except Exception:
            logger.error("Error in readHistory")
            raise Exception("Error in readHistory")
        else:
            return hist_dict

    def saveHistory(self, hist_dict: dict):
        """Write history dict to file with extension

        Args:
            hist_dict : history dictionary

        Raises:
            Exception: if error occurs

        """
        try:
            cwd = Path.cwd()
            # cwd.parents[1] for 2 levels up from current directory
            full_dir = os.path.join(cwd.parents[1], NetModel.APP_RESULTS_SUBDIR)
            file_path = Util.openFileDialog(title="Save As .json:", filters=['*.json'], dir=full_dir)
            # open file for writing
            with file_path.open("w") as f:
                # write model dict to file
                json.dump(hist_dict, f, indent=4)
        except Exception:
            logger.error("Error in saveHistory")
            raise Exception("Error in saveHistory")
