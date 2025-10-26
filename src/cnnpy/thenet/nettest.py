from keras import Sequential
#
from sklearn.metrics import confusion_matrix
#
import numpy as np
from numpy import argmax
#
import sys
from io import StringIO
from contextlib import redirect_stdout
#

#
import logging
logger = logging.getLogger()



class NetTesting:

    def __init__(self):
        self.mystdout = StringIO()

    def evaluate(self, model, x_test, y_test):
        """Evaulate network model accuracy based on test data 

        Args:
            model (Sequential) : neural network model
            x_test (ndarray): test values of images
            y_test (ndarray): test values of labels

        Raises:
            Exception: if error occurs

        Returns:
            val_loss : loss value 
            val_accuracy : accuracy value for the network model in test mode
        """

        try:
            # redirect stdout to memory string
            old_stdout = sys.stdout
            #
            with redirect_stdout(self.mystdout):
                # evaluate the model
                # loss value & metrics values for the model in test mode
                val_loss, val_accuracy = model.evaluate(x_test, y_test)
                #print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        except Exception:
            logger.error("Error in evaluate")
            raise Exception("Error in evaluate") 
        finally:
            # restore original stdout
            sys.stdout = old_stdout
            return val_loss, val_accuracy

    def predict(self, model, x_test, y_test):
        """For given training model and test data, make predictions for its classification

        Args:
            model (Sequential) : neural network model that has been trained
            x_test (ndarray): test values of images to make predictions
            y_test (ndarray): test(true) values of labels
        Raises:
            Exception: if error occurs

        Returns:
            confusion matrix (ndarray) : true values versus predicted values, with shape (n_classes by n_classes)
        """
        try:
            # redirect stdout to memory string
            old_stdout = sys.stdout
            #
            with redirect_stdout(self.mystdout):    
                #print("predict, x_test shape: ", x_test.shape)
                #  x_test.shape e.g. (10000, 28, 28, 1) for 10000 images
                pred = model.predict(x_test)
                # pred shape e.g. (10000, 10) for 10000 samples (rows) and 10 classes (cols)
                # argmax, for axis=1, returns max element in each row
                # y_pred shape, e.g. (10000,) array of predicted values (0 to 9)
                y_pred = np.argmax(pred, axis=1)
                #print("predict, y_pred shape: ", y_pred.shape)
                #
                # print("predict, y_test shape: ", y_test.shape)
                # y_test.shape, e.g. (10000,) array of true values (0 to 9)
                # cm shape e.g. (10,10) counts for each pair of true vs predicted values
                cm = confusion_matrix(y_test, y_pred)
                #print("predict, cm shape: ", cm.shape)
        except Exception as e:
            logger.error("Error in predict")
            print(f"Error in predict: {e}")
            raise Exception("Error in predict") 
        finally:
            # restore original stdout
            sys.stdout = old_stdout
            return cm
