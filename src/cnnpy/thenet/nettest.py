from keras import Sequential
#
from sklearn.metrics import confusion_matrix
#
import numpy as np
from numpy import argmax
#
import logging
logger = logging.getLogger()



class NetTesting:

    def __init__(self):
        pass

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
            #
            # evaluate the model
            # loss value & metrics values for the model in test mode
            val_loss, val_accuracy = model.evaluate(x_test, y_test)
            #print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        except Exception:
            logger.error("Error in show_samples")
            raise Exception("Error in show_samples") 
        else:    
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
            #
           # y_test_max=np.argmax(y_test, axis=0)
            y_pred = np.argmax(model.predict(x_test), axis=1)
            print("predict, x_test shape: ", x_test.shape)
            print("predict, y_pred shape: ", y_pred.shape)
            #
            cm = confusion_matrix(y_test, y_pred)
            print("predict, cm shape: ", cm.shape)
        except Exception as e:
            logger.error("Error in predict")
            print(f"Error in predict: {e}")
            raise Exception("Error in predict") 
        else:    
            return cm
