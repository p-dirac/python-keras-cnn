from keras import Sequential
from numpy import ndarray
from sklearn.model_selection import train_test_split
#
import traceback
import logging
logger = logging.getLogger()

class NetTraining:

    def __init__(self, model:Sequential, x_train:ndarray, y_train:ndarray, epochs:int, batch_size:int):
        """Train the neural network model

        Args:
            model (Sequential) : neural network model
            x_train (ndarray): training values of images
            y_train (ndarray): training values of labels
            epochs (int): number of cycles through comlete training dataset
            batch_size (int): number of samples per gradient update

        Raises:
            Exception: if error occurs

        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = batch_size
        #print("NetTraining, x_train shape: ", self.x_train.shape)
        #print("NetTraining, y_train shape: ", self.y_train.shape)
        #print("NetTraining, epochs: ", self.epochs)
        #print("NetTraining, batch_size: ", self.batch_size)

    def sizing(self, numy: int, epochs: int, batch_size: int):
        """Estimates the total number of batches, used by TrainTask to update progress bar

        Args:
            numy (int): number of training labels
            epochs (int): number of epochs to train
            batch_size (int): training batch size
        """        
        # total samples in training dataset
        self.train_sample = numy
        # number batches per epoch
        self.batches_per_epoch = self.train_sample/batch_size
        self.total_batches = self.batches_per_epoch*epochs
        print("NetTraining, train_sample: ", self.train_sample)
        print("NetTraining, batches_per_epoch: ", self.batches_per_epoch)
        print("NetTraining, total_batches: ", self.total_batches)

    def train(self, batchCallback, epochCallback):
        """Train the neural network model

        Args:
            batchCallback (function): function to update training progress
            epochCallback (function): function to update training progress

        Raises:
            Exception: if error occurs

        Returns:
            keras.callbacks.History : use hist.history to get training loss values and metrics values at successive epochs 

        """
        try:
            # set total_batches for TrainTask thread to estimate progress
            self.sizing(self.y_train.shape[0], self.epochs, self.batch_size)
            # fit the model
            hist = self.model.fit(x=self.x_train, y=self.y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=[batchCallback,epochCallback], validation_data=None, verbose=0)
        except Exception as e:
            tr = traceback.format_exc()
            logger.error(f"Error in train: {tr}")
            print(f"Error in train: {str(e)}")
            raise Exception(f"Error in train: {str(e)}") 
        else:    
            return hist

    def trainVal(self, batchCallback, epochCallback):
        """Train the neural network model with validataion

        Args:
            None
        Raises:
            Exception: if error occurs

        Returns:
            keras.callbacks.History : use hist.history to get training loss values and metrics values at successive epochs 

        """
        try:
            # split training data to reduce data size and process time
            # test_size=0.2 means 20% of x,y train dataset will be used for 
            # validation testing during model.fit()
            x_train_split, x_test_val, y_train_split, y_test_val = train_test_split(self.x_train, self.y_train, test_size=0.2)
            #
            # set total_batches for TrainTask thread to estimate progress
            self.sizing(y_train_split.shape[0], self.epochs, self.batch_size)
            # fit the model with two datasets, for training and for validation
            hist = self.model.fit(x=x_train_split, y=y_train_split, epochs=self.epochs, batch_size=self.batch_size, callbacks=[batchCallback, epochCallback], validation_data=(x_test_val, y_test_val), verbose=0)
        except Exception as e:
            print(f"Error in train: {e}")
            logger.error("Error in train")
            raise Exception("Error in train") 
        else:    
            return hist


