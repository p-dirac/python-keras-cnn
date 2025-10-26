#
from PySide6.QtCore import QRunnable, Signal, QObject
from keras.callbacks import LambdaCallback
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score, log_loss, make_scorer
from numpy import ndarray
#
import keras
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Input
from keras import Sequential
from keras.optimizers import SGD
#
from io import StringIO
from contextlib import redirect_stdout
import sys
import logging
logger = logging.getLogger()
import traceback

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Note: this class is needed in TuneTask because QRunnable does not
    support signals
    """
    finished = Signal(object)
    status = Signal(str)
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(float)


#
class TuneTask(QRunnable):
    """Subclass of QRunnable to allow runnng code in separate thread without freezing the associated GUI thread.

    Args:
        QRunnable (class): allows subclass to execute code in a separate thread by calling the run method

    Raises:
        Exception: if error occurs

    Returns:
        dict: grid search result
    """    

    #
    # class variables for static method: createTuningModel
    # hyper parameters which are not tuning parameters, but used inside createTuningModel
    NumClasses = None
    InShape = None
    DecaySteps = None
    DecayRate = None
    KernelSize = None
    MOMENTUM = None
    DROPOUT = None
    #
    StopRun = False
    MemIssue = False

    def __init__(self, num_classes: int, hyper: dict, x_train: ndarray, y_train: ndarray):
        super().__init__()

        """Tune selected parameters of the network model

        QRunnable does not inherit QObject and thus does not support signals
        Use signals from WorkerSignals which is a subclass of QObject

        Note: hyper contains user specified hyper parameters which are all used inside createTuningModel.
        The hyper parameters are divided into two subsets.
        
        Subset 1: required inside the network model, but have only a single value, and not used in the grid search
        
        Subset 2: required inside the network model, but have multiple values used in the grid search, and called tuning params

        Args:
            num_classes (int): number of dependent variables
            hyper (dict): hyper parameters for network model
            x_train (ndarray): training values of images
            y_train (ndarray): training values of labels
          
        """
        self.signals = WorkerSignals()
        #
        self.gsResult = None
        self.subtotal = 0
        self.req_stop = False
        #
        #
        self.hyper = hyper
        print("TuneTask, hyper: ", self.hyper)
        self.x_train = x_train
        self.y_train = y_train
        #
        # set class variables for use in static method: createTuningModel
        TuneTask.NumClasses = num_classes
        TuneTask.InShape = x_train.shape[1:]
        # subset 1 single value hyper parameters, which are not in the tuning params set
        TuneTask.DecaySteps = hyper["decay_steps"]
        TuneTask.DecayRate = hyper["decay_rate"]
        TuneTask.KernelSize = hyper["kernel_size"]
        TuneTask.MOMENTUM = hyper["momentum"]
        TuneTask.DROPOUT = hyper["drop_out"]
        #
        TuneTask.mystdout = StringIO()
        #
        # subset 2 hyper paramters called tuning params, which have multiple values for the grid search
        # some require a name prefix of "model__"
        self.tuning_params = self.setPrefix(hyper)
        #
        print("TuneTask, tuning_params: ", self.tuning_params)
        logger.info(f"TuneTask, tuning_params: {self.tuning_params}")
        # total samples in tune dataset
        self.tuneSamples = self.y_train.shape[0]
        # cv = 5 is known for good results, but to speed up the process, use cv = 3
        self.cv = 3
        #  total_batches: estimate of total batches to run, used by the progress bar
        self.total_batches = self.estimateBatches(self.tuning_params, self.tuneSamples, self.cv)
        print("TuneTask, total_batches: ", self.total_batches)

    def setPrefix(self, hyper):

        """From the list of hyper parameters, select those to be used in the grid search, and call them tuning parameters.
        
        Some tuning parameters are renamed with a prefix of "model__".

         Args:
            hyper (dict): hyper parameters for network model

        Returns:
            dict: dictionary of tuning parameters, some with prefix of "model__"
        """

        #
        # some require a name prefix of "model__"
        self.tuning_params = {}
        # note: .items{} returns a dict_items object, not a list
        for k, v in hyper.items():
            # select items for tuning
            if(k == "batch_size" or k == "epochs" or k == "init_rate" or k == "num_filters" or k == "num_units"):
                # check for item to add prefix
                if(k == "batch_size" or k == "epochs"):
                    # no prefix
                    self.tuning_params[k] = v
                else:
                    # reset model prefix
                    self.tuning_params[f"model__{k}"] = v
        return self.tuning_params

    def setStop(self, flag: bool):
        """Allows the user to terminate TuneTask processing at any time

        The stop flag is checked inside the createTuningModel method, and raises an exception when the flag is true. This stops the model from being created, and ends the grid search process. The GridSearchCV fit method has no built-in way of interrupting, but because it makes a new call to createTuningModel for each new combination of tuning params, the stop flag can be checked inside createTuningModel to allow stopping at discrete times within the long running process. 

        Args:
            flag (bool): true to stop processing, false to keep processing
        """        
        self.req_stop = flag
        print("TuneTask, setStop: ", self.req_stop)
        TuneTask.StopRun = flag

    def updateProgress(self):
        """Send update signal every n batches
        """  
        # update every n batches
        n = round(self.total_batches/100)
        # subtotal = number of batches processed
        if((self.subtotal % n) == 0):
            update = (self.subtotal/self.total_batches)*100
            #print("update:", update)
            #logger.info(f"TuneTask, updateProgress, subtotal: {self.subtotal}")
            #
            self.signals.progress.emit(update)

    def updateBatch(self, batch):
        """Updates the batch count, and calls updateProgress

        Args:
            batch (int): number of batches (not used here) 
        """        
        # batch count not needed, since subtotal is incremented
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

    @staticmethod
    def createTuningModel(init_rate: float, num_filters: int, num_units: int):
        """Create neural network model for tuning

        Note: this method will not work with GridSearchCV, KerasClassifier as a normal instance method, but it does work as a static method. 

        Args:
            init_rate: initial learning rate
            num_filters: number of convolution filters
            num_units: number of units in dense layer

        Returns:
            Sequential : neural network model
        """
        try:
            # clear out old models and layers from the global state
            keras.backend.clear_session()
            # define model
            model = Sequential()
            # input format (height, width, Number Channels)
            # Model needs to know the shape of input into the first layer.
            # The following layers can do automatic shape inference.
            model.add(Input(shape=TuneTask.InShape))
            #
            # convolution: to emphasize image patterns 
            # small matrix slides over an input matrix
            # multiply and sum to produce output matrix
            # 1st convolution matrix size 3 by 3 or 5 by 5
            model.add(
                Conv2D(num_filters, kernel_size=TuneTask.KernelSize, strides=1, padding="Same", activation="relu", kernel_initializer="he_uniform")
            )
            #
            # 1st pooling layer
            # selects max value from each set of overlapping filters
            # to reduce size of ouput matrix
            model.add(MaxPool2D((2, 2)))
            #
            # 2nd convolution matrix size 3 by 3 or 5 by 5
            model.add(
                Conv2D(num_filters, kernel_size=TuneTask.KernelSize, strides=1, padding="Same", activation="relu", kernel_initializer="he_uniform")
            )
            # 2nd pooling layer
            model.add(MaxPool2D((2, 2)))
            #
            # reshapes n-dimensional input tensor into a one-dimensional tensor
            model.add(Flatten())
            #
            # dense layer connects every input feature to every neuron in that layer
            # every neuron computes a weighted sum of all inputs, applies an activation function, and passes result to the next layer
            model.add(Dense(units=num_units, activation="relu", kernel_initializer="he_uniform"))
            #
            # dropout layer to increase accuracy
            # overfitting is reduced by randomly omitting half of the neurons  from the previous layer before passing to the next layer
            model.add(Dropout(TuneTask.DROPOUT))
            #
            # output layer
            # softmax normalizes output scores into probabilities that sum to one
            model.add(Dense(units=TuneTask.NumClasses, activation="softmax"))
            #
            # learning rate schedule
            # allows larger weight updates during the beginning of training
            # and smaller updates later in the training
            # Note: decay_steps is number of batches processed
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=init_rate,
                decay_steps=TuneTask.DecaySteps,
                decay_rate=TuneTask.DecayRate
            )
            #
            # define loss and optimizer
            # calculate loss function 
            # pk = predicted probability for correct class k
            # sparse_categorical_crossentropy loss: 
            #   -(1/N)sumOverN(log pki) for sum i = 1 to N, over batch size N
            #
            opt = SGD(learning_rate=lr_schedule, momentum=TuneTask.MOMENTUM)
            model.compile(
                optimizer=opt,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            if(TuneTask.StopRun is True):
                raise Exception("TuneTask, stop requested") 

        except Exception as e:
            logger.error(f"Exception in createTuningModel: {e}")
            raise Exception(f"Exception in createTuningModel: {e}")
        else:
            # model type is keras Sequential
            return model


    def run(self):
        """Tune the neural network model

        Using a list of parameters values, perform a grid search to find the best model fit that maximizes accuracy.

        Returns:
            dict: grid search result
        """
        try:
            #
            #Note: batch_size and epochs are internal to KerasClassifier, but the 
            # other hyper parameters must be args to the tuningModel
            print('TuneTask, begins')
            print("run, tuning_params: ", self.tuning_params)
            logger.info(f"tuning_params: {self.tuning_params}")
            print("run, x_train.shape: ", self.x_train.shape)
            print("run, y_train.shape: ", self.y_train.shape)
            # Check if the static method is callable
            #is_callable = callable(TuneTask.createTuningModel)
            #print(f"Is createTuningModel callable? {is_callable}")
            #
            # redirect stdout to memory string
            old_stdout = sys.stdout
            # Record the batch number at the beginning of every batch.
            batchCallback = LambdaCallback(
                on_train_batch_end=lambda batch, logs=None: self.updateBatch(batch))
            # Record epoch number at end of every epoch
            epochCallback = LambdaCallback(
                on_epoch_end=lambda epoch, logs=None: self.updateEpoch(epoch))
            #
            esti = KerasClassifier(model=TuneTask.createTuningModel, callbacks=[batchCallback,epochCallback], verbose = 2)
            #
            # cv = k: cross fold validation - divide dataset into k groups
            # using one group as test, others for training; repeat k times
            # k = 5 or 10 has been shown to yield good model performance
            # higher k values require more computation time
            #
            # n_jobs = number of CPU cores
            # n_jobs = -1 means using all processors
            # n_jobs = None means 1 CPU
            # set n_jobs = 1 to avoid pickling issue
            myscoring=["accuracy"]   #["accuracy", "neg_log_loss"]
            with redirect_stdout(TuneTask.mystdout):
                #
                gs = GridSearchCV(
                    estimator=esti, param_grid=self.tuning_params, scoring=myscoring, refit="accuracy", cv=self.cv, n_jobs = 1, error_score="raise", verbose = 2)
                #
                print('TuneTask, run, gs.fit begins')
                #
                self.gsResult = gs.fit(self.x_train, self.y_train )
            #
            #print('TuneTask, run, gs.fit done')
            #print("gs_result type: ", type(self.gsResult))
            #self.signals.result.emit(self.gsResult)
            
        except Exception as e:
            # restore original stdout
            sys.stdout = old_stdout
            #self.signals.error.emit((type(e), e, e.__traceback__))
            #tr = traceback.format_exc()
            #print(f"Error in TuneTask: {tr}")
            # don't do trace, could be user stop request
            if("out of memory" in str(e)):
                TuneTask.MemIssue = True
            print(f"Error in TuneTask: {e}")
            logger.error(f"Error in TuneTask: {e}")
        finally:
            if(TuneTask.StopRun is True):
                self.signals.status.emit("stop requested")
            elif(TuneTask.MemIssue is True):
                self.signals.status.emit("memory issue")    
            else:
                self.signals.status.emit("ok")    
            self.signals.result.emit(self.gsResult)
            self.signals.finished.emit(TuneTask.mystdout)
            # restore original stdout
            sys.stdout = old_stdout


    def estimateBatches(self, paramLists: dict, n_samples: int, cv: int):

        """Estimate number of batches to be run over all combinations of the grid parameters

        Returns:
            int: Estimated number of batches
        """        

        ba_list = paramLists["batch_size"]
        ep_list = paramLists["epochs"]
        sch_list = paramLists["model__init_rate"]
        num_sch = len(sch_list)
        f_list = paramLists["model__num_filters"]
        num_f = len(f_list)
        u_list = paramLists["model__num_units"]
        num_u = len(u_list)
        # number of cases not including batch size and epochs
        num_cases = num_sch*num_f*num_u
        num_batches = 0
        for ba in ba_list:
            # number of batches for given sample size
            num_ba = round(n_samples/ba)
            for ep in ep_list:
                # number of batches for ep epochs
                num_batches += num_ba*ep
        # multiply by number of cases not including batch size and epochs
        num_batches *= num_cases        
        # multiply by approx number of cross-validation iterations        
        num_batches *= (cv - 0.9)
        return num_batches        
