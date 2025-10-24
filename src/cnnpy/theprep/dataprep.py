import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import numpy as np
import logging
logger = logging.getLogger()
from appctrl.app_util import Util

#
class DataPrep:

    """
    Prepares data for convolution network.

    """
    # Class variables
    # default values, will be reset from prep_params
    num_classes = 1
    train_dir = "../datasets/<my_data>/training"
    test_dir = "../datasets/<my_data>/testing"
    # note: square image sizes simplify input and computations 
    # image_size: width, height of each image in input dataset
    image_size = 2
    # mode for grayscale images
    MODE_GRAY = "L"
    # mode for color images (future enhancement)
    MODE_COLOR = "RGB"
    #
    # set when image files are loaded in prepData
    x_train = None
    y_train = None 
    x_test = None
    y_test = None 

    
    # constructor function
    def __init__(self, prep_params: dict):
        self.prep_params = prep_params
        self.num_classes = prep_params["num_classes"]
        self.train_dir = prep_params["train_dir"]
        self.test_dir = prep_params["test_dir"]
        self.image_size = prep_params["image_size"]
        # mode not yet part of prep_params
        # use grayscale only
        self.mode = DataPrep.MODE_GRAY
        if(self.mode == DataPrep.MODE_GRAY):
            self.n_channels = 1
        elif(self.mode == DataPrep.MODE_COLOR):
            self.n_channels = 3     

    def loadImages(self, image_parent_dir: str, num_classes: int, image_size: int, callback):
        """Prepare input data
        Dataset directory structure should be designed like this:
        ../datasets/<group>/training/<1>
        ../datasets/<group>/training/<2>
        ../datasets/<group>/training/<3>
        ...
        ../datasets/<group>/testing/<1>
        ../datasets/<group>/testing/<2>
        ../datasets/<group>/testing/<3>

        where <group> is name of image types, such as mnist, or lung_cancer
        and <1>, <2>, <3> are folder names for each image class

        Args:
            image_parent_dir (str): parent directory name for input data,
            such as "../datasets/mnist/training", does not include sub folder
            num_classes (int): number of classes associated with the image files
            image_size (int): width or height of image, assumed to be square
            callback (function): function to update data loading progress

        Raises:
            Exception: if error occurs

        Returns:
            tuple[ndarray, ndarray]: labels as a ndarray, and features as a ndarray
        """
        try:
            # The fit function needs numpy arrays for labels, features.
            # However, lists are more efficient for appending.
            # After the lists are created, they are converted to ndarrays.
            #
            # list of image files
            featureList = []
            # list of image labels, based on index of folder order
            labelList = []
            #
            k = 0
            # image_parent_dir must be parent of k_path
            # k_path is sub-directory containing the image files
            print("loadImages image_parent_dir: ",image_parent_dir)
            # under image_parent_dir, should be only folders
            for folderName in os.listdir(image_parent_dir):
                #print("loadImages folderName: ",folderName)
                # there should be only num_classes folders
                folder = os.path.join(image_parent_dir, folderName)
                print("loadImages folder: ",folder)
                # under dir_name, should be only image files 
                for fileName in os.listdir(folder):
                    # need full path of each file for open function
                    fullPath = os.path.join(folder, fileName)
                    # check if fullPath is really a file
                    if os.path.isfile(fullPath):
                        # check file name extension
                        if ".png" in fileName:
                            imgarr = self.extractImage(fullPath, image_size)
                            # note: list append is faster than ndarray append
                            # append image to feature list
                            featureList.append(imgarr)
                            # append index of class name to label list
                            # the loss function requires a number type label
                            labelList.append(k)
                # update class name index            
                k += 1
                #print("callback k:", k)            
                callback(k)            
            # convert each list to ndarray  
            # need ndarrays for fit function later              
            features = np.asarray(featureList)
            labels = np.asarray(labelList)            
            #
            print("features shape: ",features.shape)
            print("labels shape: ",labels.shape)

        except Exception:
            print("Error in prepData")
            logger.error("Error in prepData")
            raise Exception("Error in prepData") 
        else:
            return labels, features

    def extractImage(self, fullPath, image_size):
        # mode L: 8-bit, for grayscale images.
        # mode RGB: 24 bit, for color images
        imgData = Image.open(fullPath).convert(self.mode)
        w, h = imgData.size
        #print(f"extractImage w: {w}, h: {h}")
        # check if image data is same as expected size
        if(h != image_size or w != image_size):
            # resize arg is tuple (w, h)
            # resize to square shape
            imgData = imgData.resize((image_size, image_size))
        # convert image to numpy array
        imgarr = np.array(imgData)
        # reshape to square matrix with channels?? do we need the channels ??
        # maybe for case of 3 channels
        imgarr = imgarr.reshape(image_size, image_size, self.n_channels)
        return imgarr

    @staticmethod
    def shuffleArrays(a1: np.ndarray, a2: np.ndarray, seed: int):
        """Shuffle two arrays in unison, and in-place
        This is a utility method that does not depend on the class instance
        Called by: DataPrep.shuffle(a1, a2, seed)

        Args:
            a1 (ndarray): first array to be shuffled
            a2 (ndarray): second array to be shuffled
            seed (int): used to initialize the random nuber generator
        Returns:
            None
        """
        randState = np.random.RandomState(seed)
        # Modify sequence a1 in-place by shuffling its contents.
        randState.shuffle(a1)
        #
        # use same seed so that shuffle of b is same as shuffle of a
        randState.seed(seed)
        # Modify sequence a2 in-place by shuffling its contents.
        randState.shuffle(a2)

    def transform(self, labels: np.ndarray , features: np.ndarray):
        """Transform the labels and features for processing
            
        Args:
            labels (ndarray): values of dependent variable
            features (ndarray): values of independent variable

        Raises:
            Exception: if error occurs

        Returns:
            int: 0 if no error occurs
        """
        try:
            # shuffle both labels and features together in-place
            # Note that shuffleArrays is a static method called using the class name, rather than self.
            # Also note that the argument array content can be modified without using return.
            seed = 42
            DataPrep.shuffleArrays(labels, features, seed)
            #
            # normalize the image cell values from 0 to 1
            features = features/255.0
        except Exception:
            logger.error("Error in normalize")
            raise Exception("Error in normalize") 
        else:    
            return labels, features

    def prepBothData(self, num_classes: int, image_size: int, parent_train_dir: str, parent_test_dir:str, callback):
        """Load and Transform the labels and features for processing
            
        Args:
            num_classes (int): number of possible values of dependent variable
            image_size (int): width or height of an image, assumed to be square
            parent_train_dir (str): training directory parent 
            parent_test_dir (str): testing directory parent
            callback (function): function to update data loading progress

        Raises:
            Exception: if error occurs

        Returns:
            tuple[ndarray, ndarray, ndarray, ndarray]: training features, test features, training labels, test labels

        """
        try:
            #
            # prep training data
            y_train, x_train = self.loadImages(parent_train_dir, num_classes, image_size, callback)
            y_train, x_train = self.transform(labels=y_train, features=x_train)
            #
            # prep test data
            y_test, x_test = self.loadImages(parent_test_dir, num_classes, image_size, callback)
            y_test, x_test = self.transform(labels=y_test, features=x_test)
        except Exception:
            logger.error("Error in prepBothData")
            raise Exception("Error in prepBothData") 
        else:    
            return x_train, x_test, y_train, y_test

    def prepData(self, data_dir: str, callback):
        """Load and Transform the labels and features for processing
            
        Args:
            data_dir (str): data directory parent 
            callback (function): function to update data loading progress

        Raises:
            Exception: if error occurs

        Returns:
            tuple[ndarray, ndarray, ndarray, ndarray]: features, labels

        """
        try:
            #
            # load neural network data
            y, x = self.loadImages(data_dir, self.num_classes, self.image_size, callback)
            #
            # transform data for network processing
            labels, features = self.transform(labels=y, features=x)
        except Exception:
            logger.error("Error in prepData")
            raise Exception("Error in prepData") 
        else:    
            return labels, features

    def runTask(self, data_dir: str, callback):
        """
        Run prep functions 

        Args:
            data_dir (str): data directory parent
            callback (function): function to update data loading progress

        Raises:
            Exception: if error occurs

        Returns:
            tuple[ndarray, ndarray, ndarray, ndarray]: training features, test features, training labels, test labels
        """
        try:
            print("prep, runTask")
            #
            logger.info("DataPrep, prepData begin")
            #
            # load data, shuffle, and normalize
            print("prep, image_size: ", self.image_size)
            labels, features = self.prepData(data_dir, callback)
            #
            logger.info("DataPrep, prepData finished")
        except Exception:
            logger.error("Error in prep runTask")
            raise Exception("Error in prep runTask") 
        else:
            return  labels, features