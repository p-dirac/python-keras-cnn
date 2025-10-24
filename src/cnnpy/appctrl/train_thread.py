#
from PySide6.QtCore import QThread, Signal
from keras.callbacks import LambdaCallback
#
class TrainTask(QThread):

    # Signal for the window to establish the maximum value
    # of the progress bar.
    setMaxProgress = Signal(float)
    # Signal to increase the progress.
    setUpdateProgress = Signal(float)
    #
    hist = None
    subtotal = 0

    def __init__(self, netTrain, cnnval):
        """Train the neural network model in a separate thread

        Args:
            netTrain (class): instance of NetTraining which actually performs the training by fitting the neural network to the training dataset
            cnnval (bool): true if task includes both training and validation, false if only training is desired  
        """        
        super().__init__()
        self.netTrain = netTrain
        self.cnnval = cnnval

    def updateProgress(self):
        """Send update signal every n batches
        """
        # emit update every n batches
        n = round(self.netTrain.total_batches/20)
        # subtotal = number of batches processed
        if((self.subtotal % n) == 0):
            update = (self.subtotal/self.netTrain.total_batches)*100
            #print("progress subtotal:", self.subtotal) 
            #print("progress 10*b:", 10*self.netTrain.batch_size) 
            #print("progress update:", update) 
            #
            # Note: update value must be within progress bar range
            self.setUpdateProgress.emit(update)

    def updateBatch(self, batch):
        """Updates the batch count, and calls updateProgress

        Args:
            batch (int): number of batches (not used here) 
        """        
        # batch_count not needed, since subtotal is incremented
        # on each batch
        #
        # subtotal = number of batches processed
        self.subtotal += 1
        self.updateProgress()   

    def updateEpoch(self, epoch):
        """Updates the epoch count

        Args:
            epoch (int): number of epochs (not used here)
        """   
        # epoch zero-based counter
        #print("epoch update:", epoch) 
        pass
        # epoch not needed for update
        # maybe useful in the future

    def run(self):
        """Train the neural network model

        Using a list of parameters values, perform a grid search to find the best model fit that maximizes accuracy.

        Returns:
            keras.callbacks.History: use hist.history to get training loss values and metrics values at successive epochs  
        """

        #
        # Record the batch number at the beginning of every batch.
        batchCallback = LambdaCallback(
            on_train_batch_end=lambda batch, logs=None: self.updateBatch(batch))
        # Record epoch number at end of every epoch
        epochCallback = LambdaCallback(
            on_epoch_end=lambda epoch, logs=None: self.updateEpoch(epoch))

        # make training run
        if(self.cnnval):
            # training with validation
            self.hist = self.netTrain.trainVal(batchCallback, epochCallback)
        else:
            # training only, without validation
            self.hist = self.netTrain.train(batchCallback, epochCallback)
