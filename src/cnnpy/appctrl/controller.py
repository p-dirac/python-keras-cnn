#
import os
import logging
logger = logging.getLogger()
import time
import copy
from PySide6 import QtCore
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QLayout,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGridLayout,
    QPushButton,
    QMessageBox,
    QTableWidget, 
    QTableWidgetItem, 
    QLineEdit,
    QLabel,
    QProgressBar,
    QWidget
)
import torch
import keras
from keras import Sequential
from keras.src.callbacks.history import History
from numpy import ndarray
from sklearn.model_selection import train_test_split
#
from theprep.dataprep import DataPrep
from appctrl.plotter import Plotter
from theprep.data_params import DataParams
from theprep.hyper_params import HyperParams
from thenet.netmodel import NetModel
from thenet.nettrain import NetTraining
from thenet.nettest import NetTesting
from appctrl.data_thread import DataTask
from appctrl.tune_thread import TuneTask
from appctrl.train_thread import TrainTask
from appctrl.app_util import Util
#
from PySide6.QtCore import Signal, Slot, QObject, QThreadPool
from PySide6.QtGui import QFont
from io import StringIO
#
class Controller(QObject):

    # Class variables
    # font, called like this: Controller.font, or self.font
    font = QFont("Arial", 14, QFont.Weight.Bold)
    font12 = QFont("Arial", 12, QFont.Weight.Normal)

    # send signal when dataset has been loaded
    # signal connection receiver must be set in app.py
    trainingDataIsLoaded = Signal(bool)
    testingDataIsLoaded = Signal(bool)
    updateStatus = Signal(str)
    prepComplete = Signal()
    trainingComplete = Signal()
    replaceLayout = Signal(QLayout)
    replaceWidget = Signal(QWidget)

    #
    OPTION_TRAINING = "training"
    OPTION_TESTING = "testing"

    #def __init__(self):
    def __init__(self):

        """The middleware between the GUI and the backend modules. 

        Holds instances of classes: DataParams, HyperParams, and NetModel, to allow calling methods on those classes.

        Holds data passed between classes.

        """

        # must include super call for QObject, Signal
        super().__init__()


        self.task_option = None
        #
        # paramsInstance: DataParams class instance, set in main
        self.paramsInstance = None
        #
        # data params dict
        self.params = None
        #
        # hyperInstance: HyperParams class instance, set in main
        self.hyperInstance = None
        #
        # hyper params dict
        self.hyper = None
        #
        # netModel: NetModel class instance, set in main
        self.netModel = None
        # model: keras training model, set in trainPrep
        self.model = None
        # use same seed to reproduce same results with same hyper
        keras.utils.set_random_seed(53)
        #
        # x,y features and labels, set in dataTaskFinished
        self.x_train = None
        self.y_train = None 
        self.x_test = None
        self.y_test = None 
        #
        # hist_dict set in plotHistory
        self.hist_dict = None
        # num_classes set in trainPrep
        self.num_classes = None
        #
        # true_labels = [i for i in range(self.NumClasses)]
        #
        # epochs and batch_size are used in the network model training process
        self.epochs = None
        self.batch_size = None
        #
        self.tunetask = None
        self.tune_status = None

    def __repr__(self):
        return "Controller for NetAppWin"

    def setDataParams(self, dataParams: DataParams):
        """Set instance of class DataParams

        Args:
            dataParams (class): instance of class DataParams
        """        
        # may use self.paramsInstance or Controller.paramsInstance 
        self.paramsInstance = dataParams 

    def setHyperParams(self, hyperParams: HyperParams):
        """Set instance of class HyperParams

        Args:
            hyperParams (class): instance of class HyperParams
        """        
        # may use self.hyperInstance or Controller.hyperInstance 
        self.hyperInstance = hyperParams 

    def setNetModel(self, netModel: NetModel):
        
        """Set instance of class NetModel

        Note: this is not the keras neural network model

        Args:
            netModel (class): instance of class NetModel

        """        
        self.netModel = netModel 

    def getResultDir(self):
        self.params = self.paramsInstance.getParams()
        if(self.params is None):
            Util.alert("Controller: Data Params not set.")
            return None
        result_dir = self.params["result_dir"]
        if (not os.path.exists(result_dir)):
                Util.alert(f"Results directory not found: \n {result_dir}")
                return None
        else:
            return result_dir

    @Slot()
    def initModelDir(self):
        """Called when Data Params are loaded to set Results directory in netModel
        """        
        self.result_dir = self.getResultDir()
        if(self.result_dir is not None):
            self.netModel.setResultDir(self.result_dir)


    def prepTrainingData(self):

        """Calls DataPrep and DataTask to load the training dataset.

        DataTask runs in a separate thread to avoid blocking the GUI.
        The method dataTaskFinished is called when loading has completed, and the samples method plots a few samples to verify the data. 
        """        
        
        #print("netcontrol, prep")
        self.params = self.paramsInstance.getParams()
        if(self.params is None):
            Util.alert("Controller: Data Params not set.")
            return
        #print("netcontrol, data params: ", self.params)    
        #
        data_dir = self.params["train_dir"]
        if (not os.path.exists(data_dir)) :
                Util.alert(f"Training image directory not found: \n {data_dir}")
                return
        #
        self.updateStatus.emit("Loading training datasets")
        dataPrep = DataPrep(self.params)
        # Pass dataPrep to thread task, and call dataPrep.runTask 
        # inside the thread. The runTask method will call prepData.
        dataBar = self.anyBar("Loading Training Datasets")
        self.datatask = DataTask(dataPrep, data_dir)
        self.datatask.setObjectName("Data Task") 
        #
        # prepare Signal connections
        self.datatask.setMaxProgress.connect(dataBar.setMaximum)
        self.datatask.setUpdateProgress.connect(dataBar.setValue)
        #  after longtask is completed, it will call dataTaskFinished
        self.datatask.finished.connect(lambda: self.dataTaskFinished(task_option=Controller.OPTION_TRAINING))
        # begin separate thread for loading data
        self.datatask.start()
        # must not do anything after start

    def prepTestingData(self):
        """Calls DataPrep and DataTask to load the testing dataset.

        DataTask runs in a separate thread to avoid blocking the GUI.
        The method dataTaskFinished is called when loading has completed, and the samples method plots a few samples to verify the data.  
        """        
        #print("netcontrol, prepTestingData")
        self.params = self.paramsInstance.getParams()
        if(self.params is None):
            Util.alert("Controller: Data Params not set.")
            return
        print("netcontrol,  data params: ", self.params)
        #
        # pass dataPrep to thread task, and call dataPrep.runTask 
        # inside the thread
        data_dir = self.params["test_dir"]
        if (not os.path.exists(data_dir)) :
                Util.alert(f"Testing image directory not found: \n {data_dir}")
                return
        #
        self.updateStatus.emit("Loading test datasets")
        dataPrep = DataPrep(self.params)
        dataBar = self.anyBar("Loading Test Datasets")
        self.datatask = DataTask(dataPrep, data_dir)
        self.datatask.setObjectName("Data Task") 
        #
        # prepare Signal connections
        self.datatask.setMaxProgress.connect(dataBar.setMaximum)
        self.datatask.setUpdateProgress.connect(dataBar.setValue)
        #  after longtask is completed, it will call dataTaskFinished
        self.datatask.finished.connect(lambda: self.dataTaskFinished(task_option=Controller.OPTION_TESTING))
        # begin separate thread for loading data
        self.datatask.start()
        # must not do anything after start

    def samples(self, labels: ndarray, features: ndarray):
        """Plots a few image samples to verify that loading worked properly.

        Args:
            labels (ndarray): values of labels
            features (ndarray): values of images
        """  
        try:      
            # Canvas setup, Plotter is subclass of Qwidget
            #
            central = QWidget()
            self.replaceWidget.emit(central)
            #
            # vertical layout for display
            data_layout = QVBoxLayout()
            central.setLayout(data_layout)
            # add Plotter 
            plotter = Plotter()
            data_layout.addWidget(plotter)
            #
            hbox = QWidget()
            hlayout = QHBoxLayout(hbox)
            hlayout.setAlignment(Qt.AlignCenter) 
            #
            okbtn = QPushButton("OK")
            okbtn.setFixedWidth(200)
            okbtn.setStyleSheet("background-color: #E0FFFF; border-radius: 5px; border-color: #DCDCDC; border-width: 2px; border-style: outset;")
            okbtn.setFont(self.font)
            hlayout.addWidget(okbtn)
            #
            # connect okbtn button to function
            okbtn.clicked.connect(self.closeCentral)
            #
            # add ok button
            data_layout.addWidget(hbox)

            #print("Controller.samples plotting") 
            self.updateStatus.emit("Plotting samples")
            #print("labels shape: ", labels.shape)
            #
            if((labels is not None) and (features is not None)):
                #print("Controller.samples calling plotSamples") 
                plotter.plotSamples(labels, features)
                #plotter.update()
                self.updateStatus.emit("Samples plotted") 
            else:
                print("Error in Controller.samples, labels or features is None")
                logger.error("Error in Controller.samples, labels or features is None")
        except Exception as e:
            print(f"Controller.samples raised an exception: {e}")
            logger.error(f"Controller.samples raised an exception: {str(e)}")
            raise Exception(f"Controller.samples exception: {str(e)}") 


    def runTuner(self):
        """Runs the TuneTask in a separate thread, and calls extractTuneResults and tuneTable to display the grid search results.
        """ 
        try:       
            self.params = self.paramsInstance.getParams()
            if(self.params is None):
                Util.alert("Controller: Data Params not set.")
                return
            self.hyper = self.hyperInstance.getHyper()
            #
            if(self.hyper is None):
                Util.alert("Controller: Hyper Params not set.")
                return
            self.num_classes = self.params["num_classes"]
            if(self.netModel is None):
                Util.alert("Controller: netModel not set.")   
                return 
            #
            # clear out old models
            self.clearKeras()
            #print(torch.cuda.memory.memory_summary(device=None, abbreviated=False))
            #
            y_size = self.y_train.shape[0]
            print("y_size: ", y_size)
            if(y_size > 10000):
                # train_size=10000 means only a fraction of x,y train dataset will be used for x_tune, y_tune during model.fit() tuning 
                self.x_tune, img_test, self.y_tune, lbl_test = train_test_split(self.x_train, self.y_train, train_size=10000)
            else:
                # otherwise use the whole dataset
                self.x_tune = self.x_train
                self.y_tune = self.y_train  
            #      
            tune_size = self.y_tune.shape[0]
            print("tune_size: ", tune_size)
            #
            tuneBar = self.anyBar("Tuning in Progress \n (May take several minutes)")
            #
            # Note: self.hyper should contain both the normal hyper parameters, and the lists of model tuning parameters
            self.tunetask = TuneTask(self.num_classes, self.hyper, self.x_tune, 
            self.y_tune)
            self.tunetask.signals.status.connect(self.tuneStatus)
            self.tunetask.signals.result.connect(self.extractTuneResults)
            self.tunetask.signals.finished.connect(self.tuneTaskFinished)
            self.tunetask.signals.progress.connect(tuneBar.setValue)
            #worker.signals.error.connect(self.task_error)
            #
            # Execute the worker in a separate thread
            self.threadpool = QThreadPool()
            self.threadpool.start(self.tunetask)
        except Exception as e:
            print(f"runTuner raised an exception: {e}")
            logger.error(f"runTuner raised an exception: {str(e)}")
            raise Exception(f"runTuner exception: {str(e)}") 


    def stopTuner(self):
        self.req_stop = True 
        if(self.tunetask is not None):
            self.tunetask.setStop(self.req_stop)

    def trainPrep(self):

        """Prepare for training of neural network

        Prerequisites: params, hyper, x_train

        Creates keras neural network training model from hyper parameters

        Plots model summary table to show layers, shapes, and sizing info.
        """  
        try:      
            #print("netcontrol, trainPrep")
            self.params = self.paramsInstance.getParams()
            self.hyper = self.hyperInstance.getHyper()
            #
            if(self.params is None):
                Util.alert("Controller: Data Params not set.")
                return
            #
            if(self.hyper is None):
                Util.alert("Controller: Hyper Params not set.")
                return
            if(self.netModel is None):
                Util.alert("Controller: netModel not set.")   
                return 
            #
            # access dict params using key in [] and first entry in list [0]
            self.epochs = self.hyper["epochs"][0]
            self.batch_size = self.hyper["batch_size"][0]
            print("netcontrol.trainPrep, epochs: ", self.epochs, ", batch_size: ", self.batch_size)
            #
            #print("netcontrol, x_train shape: ", self.x_train.shape)
            in_shape = self.x_train.shape[1:]
            #print("netcontrol, x_train input shape: ", in_shape)
            # access prepParams, the num_classes
            self.num_classes = self.params["num_classes"]
            self.result_dir = self.params["result_dir"]
            self.netModel.setInfo(in_shape, self.num_classes, self.result_dir, self.hyper)
            #
            # create keras neural network training model
            self.model = self.netModel.createModel()
            # to print
            # self.model.summary()
            #
            self.showSummary(self.model)
        except Exception as e:
            print(f"Controller.trainPrep raised an exception: {e}")
            logger.error(f"Controller.trainPrep raised an exception: {str(e)}")
            raise Exception(f"Controller.trainPrep exception: {str(e)}") 

    def showSummary(self, model: Sequential):

        """Creates table of network model layers, shapes, and sizing info.

        Args:
        model (class): network model

        """  
        try:      
            # extract summary data from the model
            a = self.extractSummary(model)
            nrows = len(a)
            ncols = len(a[0])
            #
            #print("showSummary nrows: ", nrows, ", ncols: ", ncols)
            central = QWidget()
            self.replaceWidget.emit(central)
            #
            # vertical layout for display
            summary_layout = QVBoxLayout()
            central.setLayout(summary_layout)
            #
            #print("showSummary heading: ", self.heading)
            heading = QLabel(self.heading, alignment=Qt.AlignHCenter)
            summary_layout.addWidget(heading)
            #
            model_table = QTableWidget()
            summary_layout.addWidget(model_table)
            #
            # do not use alignment on QTableWidget - the table will shrink!
            #summary_layout.addWidget(model_table, alignment=Qt.AlignCenter)
            #
            # hide row numbering
            model_table.verticalHeader().setVisible(False)
            # hide column numbering
            model_table.horizontalHeader().setVisible(False)
            #
            model_table.setRowCount(nrows)
            model_table.setColumnCount(ncols)
            # w = col widths
            w = [500, 300, 200]
            for j in range(ncols):
                # set width of col j
                model_table.setColumnWidth(j, w[j])
            #colHdr e.g. ["A", "B", "C" ]
            #table.setHorizontalHeaderLabels(colHdr)
            for i in range(nrows):
                for j in range(ncols):
                    model_table.setItem(i, j, QTableWidgetItem(a[i][j]))
            vtail = QVBoxLayout()
            vtail.setAlignment(Qt.AlignHCenter) 
            #
            # tail
            for t in self.tail:
                tail = QLabel(t, alignment=Qt.AlignLeft)
                vtail.addWidget(tail)
            summary_layout.addLayout(vtail)    
            #
            # horizontal layout for Ok button
            hlayout = QHBoxLayout()
            hlayout.setAlignment(Qt.AlignCenter) 
            #
            okbtn = QPushButton("OK")
            okbtn.setFixedWidth(200)
            okbtn.setStyleSheet("background-color: #E0FFFF; border-radius: 5px; border-color: #DCDCDC; border-width: 2px; border-style: outset;")
            okbtn.setFont(self.font)
            hlayout.addWidget(okbtn)
            #
            # connect okbtn button to function
            okbtn.clicked.connect(self.sendPrepSignal)
            okbtn.clicked.connect(self.closeCentral)
            #
            summary_layout.addSpacing(20)
            summary_layout.addLayout(hlayout)
            summary_layout.addStretch(0) 
        except Exception as e:
            print(f"Controller.showSummary raised an exception: {e}")
            logger.error(f"Controller.showSummary raised an exception: {str(e)}")
            raise Exception(f"Controller.showSummary exception: {str(e)}") 

    def extractSummary(self, model: Sequential):

        """Creates array of network model specs

        First extract all lines from model summary. Skip every other line to retreive actual rows of data items. Call extractColumns to form array of data items.

        Args:
        model (class): network model

        Returns:
        array: two dimensional array of network model specs
        """        
        # extract each line based on the newline character
        self.strlist = []
        model.summary(print_fn=lambda x: self.extractLines(x))
        n = len(self.strlist)
        # save 1st line
        self.heading = self.strlist[0]
        #print("strlist, heading: ", self.heading)
        # save last 3 lines
        self.tail = self.strlist[-4:n-1]
        #print("strlist, tail: ", self.tail)
        # from start, skip every other line
        rows = self.strlist[0:n-4:2]
        nrows = len(rows)
        # rows includes heading, omit in call to extractColumns
        a = self.extractColumns(rows[1:])
        return a

    def extractLines(self, x: str):
        """Splits a string into separate lines

        Args:
            x (str): _description_
        """        
        self.strlist=x.split("\n")

    def extractColumns(self, rows: list):

        """Creates array of network model specs

        From each row, split at unicode character, then extract column item and add to array.

        Args:
        rows (list): network model summary rows

        Returns:
        array: two dimensional array of network model specs
        """        
        # n = number of rows
        n = len(rows)
        #s = rows[1]
        #for c in s:
        #    print("extract, c: ", c, ", uni: ", ord(c))
        w = 3 # width, number of columns 
        a = [[0 for x in range(w)] for y in range(n)]
        #
        i = 0
        sep74 = chr(9474)
        sep75 = chr(9475)
        for row in rows:
            if(i > 0):
                sp = row.split(sep74)
            else:
                sp = row.split(sep75)
            v = len(sp)
            # sep at beginning and end of string causes '' characters
            # skip 1st and last empty char ''
            items = sp[1:v-1]
            #print("extract, items: ", items)
            #print("extract, items len: ", len(items))
            j = 0
            for item in items:
                a[i][j] = item.strip()
                # next column
                j += 1
            # next row    
            i += 1    
        return a

    def sendPrepSignal(self):
        # send signal to enable menu buttons
        self.prepComplete.emit()

    def closeCentral(self):
        # send signal to replace central widget with empty widget
        central = QWidget()
        self.replaceWidget.emit(central)

    def train2(self, include_val):

        """Run the training task on a separate thread

        While the task is running, a progress bar is displayed to the user.

        Args:
            include_val (bool): true if task includes both training and validation, false if only training is desired    
        """  
        try:      
            #
            print("netcontrol, start training")
            if(self.y_train is None):
                Util.alert("Training dataset is not loaded.")
                return
            if(self.batch_size is None):
                Util.alert("Batch size is not set.")
                return
            if(self.model is None):
                Util.alert("Controller: model not set.")
                return

            #print("netcontrol, y_train type: ", type(self.y_train))
            print("netcontrol, x_train shape: ", self.x_train.shape)
            print("netcontrol, batch_size: ", self.batch_size)
            #
            # clear out old models
            self.clearKeras()
            #
            netTrain = NetTraining(self.model, self.x_train, self.y_train, self.epochs, self.batch_size)
            #
            self.updateStatus.emit("Training underway")

            netBar = self.anyBar("Neural Network running \n (May take several minutes)")
            #
            # pass netTrain to thread task, and call netTrain.train 
            # inside the thread
            self.traintask = TrainTask(netTrain, include_val)
            #
            # prepare Signal connections
            self.traintask.setMaxProgress.connect(netBar.setMaximum)
            self.traintask.setUpdateProgress.connect(netBar.setValue)
            #  after task is completed, it will call trainTaskFinished
            # using lambda as no arg function for connect
            # which allows the actual function to pass an argument
            self.traintask.finished.connect(lambda: self.trainTaskFinished(include_val))
            # begin separate thread for network training run
            self.traintask.start()
            # must not do anything after start
        except Exception as e:
            print(f"Controller.train2 raised an exception: {e}")
            logger.error(f"Controller.train2 raised an exception: {str(e)}")
            raise Exception(f"Controller.train2 exception: {str(e)}") 

    def anyBar(self, title: str):
        """Create progress bar

        Args:
            title (str): title of progress bar

        Returns:
            class: progress bar
        """        
        label = QLabel(title, alignment=Qt.AlignHCenter)
        myBar = QProgressBar()
        myBar.setRange(0, 100)
        myBar.setValue(0)
        # set the format to display percentage 
        myBar.setFormat("%p%") 
        # alignment of percentage text within bar
        myBar.setAlignment(QtCore.Qt.AlignHCenter) 
         # changing the color of process bar
        myBar.setStyleSheet("QProgressBar::chunk "
                "{"
                "background-color: lightcyan;"
                "}"
                "QProgressBar"
                "{"
                "border: 2px solid #DCDCDC;"
                "border-radius: 4px;"
                "text-align: center;"   
                "}")

        myBar.setFixedSize(500,50)
        #
        # Layout setup
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # add top spacing
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addSpacing(70)
        layout.addWidget(myBar)
        # add bottom spacing
        layout.addStretch(2)
        #
        self.replaceLayout.emit(layout)
        return myBar


    def predict(self):

        """Uses test dataset to predict results based on a trained network model (a Keras Sequential object)

        Results are displayed as the confusion matrix, which shows how many predicted labels matched the true labels.

        Raises:
            Exception: if error occurs
        """        

        try:
            if(self.y_test is None):
                Util.alert("Testing dataset is not loaded.")
                return
            if(self.params is None):
                Util.alert("Data params is not loaded.")
                return
            if(self.netModel is None):
                Util.alert("NetModel is not set.")
                return

            # model is a Sequential
            self.model = self.netModel.getModel()
            if(self.model is None):
                Util.alert("Model is not loaded or prepared.")
                return
            #
            self.updateStatus.emit("Prediction underway")
            # need a message box to allow updateStatus signal
            Util.autoCloseMessageBox(1000, "Prediction underway")
            #btn = QMessageBox.information(None, "Predict", "Prediction underway")
            # clear out old models
            self.clearKeras()
            #
            self.testing = NetTesting() 
            cm = self.testing.predict(self.model, self.x_test, self.y_test)
            #
            self.plotConfusion(cm)
        except Exception as e:
            print(f"predict raised an exception: {e}")
            logger.error(f"predict raised an exception: {str(e)}")
            raise Exception(f"Error in predict: {str(e)}") 


    def plotConfusion(self, cm: ndarray):
        """Plot confusion matrix for test dataset

        Args:
            cm(): confusion matrix
        Raises:
            Exception: if error
        """        
        try:
            #
            # note: Plotter is subclass of Qwidget
            plotter = Plotter()
            # for plot, for label list use index
            self.num_classes = self.params["num_classes"]
            classLabels = [i for i in range(self.num_classes)]
            plotter.plotConfusion(cm, classLabels)
            #
            central = QWidget()
            #
            # vertical layout for display
            predict_layout = QVBoxLayout()
            # add Plotter 
            predict_layout.addWidget(plotter)
            #
            central.setLayout(predict_layout)
            self.replaceWidget.emit(central)
            #
            hbox = QWidget()
            hlayout = QHBoxLayout(hbox)
            # center the button
            hlayout.setAlignment(Qt.AlignHCenter) 
            #
            okbtn = QPushButton("OK")
            okbtn.setFixedWidth(200)
            okbtn.setStyleSheet("background-color: #E0FFFF; border-radius: 5px; border-color: #DCDCDC; border-width: 2px; border-style: outset;")
            okbtn.setFont(self.font)
            hlayout.addWidget(okbtn)
            #
            # connect okbtn button to function
            okbtn.clicked.connect(self.closeCentral)
            predict_layout.addSpacing(20)
            #
            # add ok button
            predict_layout.addWidget(hbox)
            predict_layout.addStretch(0) 
        except Exception as e:
            print(f"predict raised an exception: {e}")
            logger.error(f"predict raised an exception: {str(e)}")
            raise Exception(f"Error in predict: {str(e)}") 


    def evaluate(self):

        """Uses test dataset to evaluate overall scores based on a trained network model.

        Two scores are displayed: accuracy and loss.
        Accuracy is the total number of correctly predicted labels as a percentage of the total number of test samples.
        Loss is the sparse categorical crossentropy calculation.

        Raises:
            Exception: if error occurs
        """        

        try:
            if(self.y_test is None):
                Util.alert("Testing dataset is not set.")
                return
            if(self.params is None):
                Util.alert("Data params is not set.")
                return
            if(self.netModel is None):
                Util.alert("NetModel is not set.")
                return

            self.model = self.netModel.getModel()
            if(self.model is None):
                Util.alert("Model is not loaded or prepared.")
                return
            #
            self.updateStatus.emit("Evaluation underway")
            # need a message box to allow updateStatus signal
            Util.autoCloseMessageBox(1000, "Evaluation underway")
            # clear out old models
            self.clearKeras()
            #
            self.testing = NetTesting() 
            loss, accuracy = self.testing.evaluate(self.model, self.x_test, self.y_test)
            #
            self.evalTable(accuracy, loss)

        except Exception as e:
            print(f"evaluate raised an exception: {e}")
            logger.error(f"evaluate raised an exception: {str(e)}")
            raise Exception(f"Error in evaluate: {str(e)}") 

    def evalTable(self, accuracy: float, loss: float):
        """Create small table to dislay overall accuracy and loss

        Args:
            accuracy (float): model accuracy
            loss (float): model loss
        """
        try:
            central = QWidget()
            #
            # vertical layout for display
            eval_layout = QVBoxLayout()
            eval_layout.setAlignment(Qt.AlignHCenter)
            central.setLayout(eval_layout)
            #
            self.replaceWidget.emit(central)
            #
            heading = QLabel("Evaluation", alignment=Qt.AlignHCenter)
            eval_layout.addWidget(heading)
            eval_layout.addSpacing(20)
            #
            acc_lbl = QLabel("Accuracy", alignment=Qt.AlignLeft)
            acc_val = QLabel(f"{100*accuracy:.2f}%", alignment=Qt.AlignLeft)
            loss_lbl = QLabel("Loss", alignment=Qt.AlignLeft)
            loss_val = QLabel(f"{loss:.3f}", alignment=Qt.AlignLeft)
            #
            # must set max height to compact the grid
            acc_lbl.setMaximumHeight(32)
            acc_val.setMaximumHeight(32)
            #
            loss_lbl.setMaximumHeight(32)
            #
            loss_val.setMaximumHeight(32)
            #
            gbox = QWidget()
            gbox.setFixedSize(500,200)
            gbox.setObjectName("mygrid")
            #
            # this StyleSheet does not work
            central.setStyleSheet
            ("""
            QWidget#mygrid
            {
            background-color: #74D4FF; border-radius: 5px; border-color: #DCDCDC; border-width: 3px; border-style: outset;
            }
            """)
            #
            # this StyleSheet does not work
            #gbox.setStyleSheet
            ("""
            QWidget#mygrid
            {
            background-color: #74D4FF; border-radius: 5px; border-color: #DCDCDC; border-width: 3px; border-style: outset;
            }
            """)

            #
            glayout = QGridLayout()
            glayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
            glayout.addWidget(acc_lbl, 0, 0)
            glayout.addWidget(acc_val, 0, 1)
            glayout.addWidget(loss_lbl, 1, 0)
            glayout.addWidget(loss_val, 1, 1)
            #
            # left, top, right, bottom
            glayout.setContentsMargins(50, 30, 50, 30) 
            glayout.setHorizontalSpacing(30)
            glayout.setVerticalSpacing(20)
            glayout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            # 
            gbox.setLayout(glayout)      
            eval_layout.addWidget(gbox) 
            #
            #
            hbox = QWidget()
            hlayout = QHBoxLayout()
            hbox.setLayout(hlayout)
            # center the button
            hlayout.setAlignment(Qt.AlignHCenter) 
            #
            okbtn = QPushButton("OK")
            okbtn.setFixedWidth(200)
            okbtn.setStyleSheet("background-color: #E0FFFF; border-radius: 5px; border-color: #DCDCDC; border-width: 2px; border-style: outset;")
            okbtn.setFont(self.font)
            hlayout.addWidget(okbtn)
            #
            # connect okbtn button to function
            okbtn.clicked.connect(self.closeCentral)
            eval_layout.addSpacing(20)
            #
            # add ok button
            eval_layout.addWidget(hbox)
            # stretch adds vertical space after gbox
            eval_layout.addStretch()
        except Exception as e:
            print(f"evaluate raised an exception: {e}")
            logger.error(f"evaluate raised an exception: {str(e)}")
            raise Exception(f"Error in evaluate: {str(e)}") 

    def plotHistory(self, hist: dict):
        """Plot the epoch history of accuracy and loss, based on just training data, without any validation.

        Args:
            hist (dict): history produced from training the network model
        """        
        # Plotter is subclass of FigureCanvas
        plotter = Plotter()
        # add plot to main window
        self.replaceWidget.emit(plotter)
        #print("netcontrol, plot the history accuracy, loss")
        plotter.plotHistory(hist)

    def plotHistoryVal(self, hist: History):
        """Plot the epoch history of accuracy and loss for both training data and validation data.

        Args:
            hist (History): history produced from training the network model, with validation data
        """        
        # Canvas setup
        plotter = Plotter()
        self.replaceWidget.emit(plotter)
        print("netcontrol, plot the history accuracy, loss")
        ax = plotter.plotHistoryVal(hist)

    def dataTaskFinished(self, task_option: str):
        """Plots a few samples after loading a dataset of images

        Args:
            task_option (str): indicates whether the task was loading training or testing data.
        """     
        try:   
            features = self.datatask.features
            labels = self.datatask.labels
            print("datatask task_option:", task_option)
            #print("datatask before quit, is running?:", self.datatask.isRunning())
            # thread finished
            self.datatask.quit()
            #print("datatask after quit, is running?:", self.datatask.isRunning())
            if(not self.datatask.isRunning):
                # clear thread memory
                self.datatask.deleteLater()
            # print a few cells
            if(features is not None):
                btn = QMessageBox.question(
                    None,
                    "Question",
                    "Show a few samples?",
                )
                if btn == QMessageBox.Yes:
                    #cells = self.features[1, 14, 1:, 0]
                    #print("dataTaskFinished, features cells:", cells)
                    #
                    # plot a few samples
                    self.samples(labels, features)
                else:
                    layout = QVBoxLayout()
                    self.replaceLayout.emit(layout)    
            #
            if(task_option is Controller.OPTION_TRAINING):
                # set x,y train
                self.x_train = features
                self.y_train = labels 
                # send signal to enable task actions
                self.trainingDataIsLoaded.emit(True)
            elif(task_option is Controller.OPTION_TESTING): 
                # set x,y test   
                self.x_test = features
                self.y_test = labels 
                # send signal to enable task actions
                self.testingDataIsLoaded.emit(True)
        except Exception as e:
            print(f"dataTaskFinished raised an exception: {e}")
            logger.error(f"dataTaskFinished raised an exception: {str(e)}")
            raise Exception(f"Error in dataTaskFinished: {str(e)}")


    def tuneStatus(self, status: str):
        """When tuning task has finished, receive status

        Args:
            status (str): either: stop requested, memory issue, or ok
        """    
        self.tune_status = status
        #print('tuneStatus signal: status: ', self.tune_status)

    def tuneTaskFinished(self, midpt: StringIO):
        """When tuning task has finished, check tune status.
        
        If status is ok, do nothing.
        If status 'stop requested' or 'memory issue', convert stdout lines to table format.

        Args:
            midpt (StringIO): stdout lines recorded while tuning was running
        """        
        #print('TuneTask, tunetask signal: status: ', self.tune_status)
        if((self.tune_status is None) or (self.tune_status == "ok")):
            print('TuneTask, tunetask finished successfully')
        elif(self.tune_status == "stop requested"):
            # Convert to table format if 'stop requested', in which
            # case tune result will be None, but midpt will be set.
            self.tuneTaskPartial(midpt)    
        elif(self.tune_status == "memory issue"):
            # Convert to table format if 'memory issue', in which
            # case tune result will be None, but midpt will be set.
            self.tuneTaskPartial(midpt)    
            
    def tuneTaskPartial(self, midpt: StringIO):
        """When tuning task has been stopped by request, convert stdout lines to table format.

        Args:
            midpt (StringIO): stdout lines recorded while tuning was running, but interrupted by stop signal
        """        
        print('TuneTask, tunetask stopped')
        midval = midpt.getvalue()
        if(midval is not None):
            # midval is a long string consisting of lines ending in a newline character
            # split the long string into individual lines
            lines = self.getLines(midval)
            nlines = len(lines)
            print("nlines: ", nlines)
            print("lines 1: ", lines[1])
            if(nlines > 0):
                # convert stdout lines into rows of param values
                rows = self.partial(lines)
                nrows = len(rows)
                print("rows 0: ", rows[0])
                if(nrows > 0):
                    # display row parameters in table format
                    self.partialTable(rows)
                else:
                    print("tuneTaskFinished, nrows : ", nrows)    
        else:
            print("tuneTaskFinished error: midval is None")
            logger.error("tuneTaskFinished error: midval is None")
            

    def partialTable(self, rows: list):

        """Create table to display partial grid search results

        Args:
            rows (list): list of params, accuracy, and loss values

        Raises:
            Exception: if error occurs
        """        
        try:
            table = QTableWidget()
            central = QWidget()
            self.replaceWidget.emit(central)
            layout = QVBoxLayout()
            central.setLayout(layout)
            # Create table title
            title = QLabel("Partial Grid Search Results")
            title.setAlignment(Qt.AlignHCenter) 
            title.setStyleSheet("font-size: 20px; font-weight: bold;")
            layout.addWidget(title) 
            # do not use alignment on QTableWidget - the table will shrink! 
            layout.addWidget(table)      
            #
            table.verticalHeader().setVisible(False)
            #table.horizontalHeader().setVisible(False)
            #
            nrows = len(rows)
            table.setRowCount(nrows)
            #
            # each item in rows is a dict of tune param:value pairs
            # e.g. {'batch_size': '64', 'epochs': '5', 'model__init_rate': '0.02', 'model__num_filters': '32', 'model__num_units': '512', 'accuracy': '0.9898', 'loss': '0.0324'}
            # use the first item in rows to get the number of pairs
            # each param will be a separate col
            mcols = len(rows[0])
            table.setColumnCount(mcols)
            #
            # The keys of the param:value pairs will be the table col headers.
            # Note: .keys() function returns a dict_keys, which looks like a list, but is not a list. It must be converted to a list as follows.
            colHdr = list(rows[0].keys())
            # remove the param prefix "model__"
            for col in range(mcols):
                table.setColumnWidth(col, 150)
                if(colHdr[col].startswith("model__")):
                    colHdr[col] = colHdr[col].replace("model__", "")
            #     
            table.setHorizontalHeaderLabels(colHdr)
            #
            for k in range(nrows):
                # note:
                # .values() returns a dict_values view of param:value pairs
                # it must be converted to a list as follows.
                col_vals = list(rows[k].values())
                # append mean value, neg loss to last 2 columns
                mean = float(col_vals[-2])
                # format % float to two decimal places
                m = "{:.2f}".format(mean*100)
                col_vals[-2] = m
                #loss = col_vals[-1]
                #print("midTable col_vals", col_vals)
                for col in range(mcols):
                    # set each col value in current row
                    val = col_vals[col]
                    table.setItem(k, col, QTableWidgetItem(str(val)))
        except Exception as e:
            print(f"partialTable raised an exception: {e}")
            logger.error(f"partialTable raised an exception: {str(e)}")
            raise Exception(f"Error in partialTable: {str(e)}")

    def partial(self, lines: list):
        """Convert stdout lines into rows of dictionary objects

        Args:
            lines (list): lines from stdout

        Returns:
            list: each row contains dictionary of params, accuracy, and loss in form of key-value pairs
        """  
        try:      
            # cvlist of tuples (i,x)
            # the [CV] line contains tuning param keys and values
            #
            # e.g. [CV] END batch_size=64, epochs=5, model__init_rate=0.02, model__num_filters=32, model__num_units=512; total time=   5.1s
            # use enumerate to get both the index and the value of each line
            cvlist = [(i,x) for i,x in enumerate(lines) if x.startswith("[CV]") ]
            rows = list()
            # each cv tuple in list start with [CV]
            for cv in cvlist:
                score = dict()
                # get index of first [CV] line
                i = cv[0]
                #
                # get prev line at i - 2, which contains the accuray and loss
                # e.g. 150/150 - 1s - 6ms/step - accuracy: 0.9892 - loss: 0.0336
                accline = lines[i-2]
                #
                # get index of accuracy substring
                accbeg = accline.find("accuracy: ")
                acclen = len("accuracy: ")
                # keep 6 digits like 0.9892 or 0.0336
                acc = accline[accbeg+acclen:accbeg+acclen+6]
                #
                # get index of loss substring
                lossbeg = accline.find("loss: ")
                losslen = len("loss: ")
                loss = accline[lossbeg+losslen:lossbeg+losslen+6]
                #
                # get line valu, which starts with [CV] END
                pline = cv[1]
                semi = pline.find(";")
                beg = len("[CV] END ")
                #extract substring after the "[CV] END " up to the ";"
                p = pline[beg:semi]
                # split substring to separate key=value pairs
                plist = p.split(", ")
                for item in plist:
                    # get key, value as list
                    kv = item.split("=")
                    # pkey is parameter name
                    pkey = kv[0]
                    # pval is parameter value
                    pval = kv[1]
                    # store key, value in score dict
                    score[pkey] = pval
                # also accuracy and loss in in score dict    
                score["accuracy"] = acc
                score["loss"] = loss
                # store score dict in rows list
                rows.append(score)
               
        except Exception as e:
            print(f"partial raised an exception: {e}")
            logger.error(f"partial raised an exception: {str(e)}")
            raise Exception(f"Error in partial: {str(e)}")
        else:
            return rows

    def getLines(self, s: str):
        """Splits a string into separate lines

        Args:
            s (str): string with multiple lines
        Returns:
            list: separate lines    
        """        
        lines = s.split("\n")
        return lines

    def extractTuneResults(self, result: dict):
        """When the tune task has completed successfully, create a table of grid search results.

        The columns are labeled with the names of the hyper parameters, with the last column being the mean accuracy %.

        If there is no result, the window may display partial results(see tuneTaskPartial). This may occur when the user selects the stop tuning menu item.

        Args:
            result (dict): grid search result
        """        
        #print('TuneTask, extractTuneResults')
        if(result is not None):
            means = result.cv_results_['mean_test_accuracy']
            #losses = result.cv_results_['mean_test_neg_log_loss']
            params = result.cv_results_['params']
            #print("gsResult", result)
            # result.best_params_: dict
            # e.g. {'batch_size': 64, 'epochs': 10, 'model__init_rate': 0.02, 'model__num_filters': 32, 'model__num_units': 700}
            best = result.best_params_
            score = result.best_score_
            print("Best params", result.best_params_)
            # result.best_score_: float, e.g. 0.9788996296
            print("Best score", result.best_score_)
            logger.info(f"Best params: {result.best_params_}")
            logger.info(f"Best score: {100*result.best_score_:,.2f}%")
            self.tuneResults(means, params, best, score)

        else:
            print("extractTuneResults error: result is None")
            logger.error("extractTuneResults error: result is None")
            self.closeCentral()

    def tuneResults(self, means: list, params: list, best: dict, score: float):
        """Create panel to display grid search results

        Args:
            means (list): list of mean accuracy values
            params (list): list of dict for each set of grid search parameters
            best (dict): dict of parameters associated with best score
            score (float): score for best parameters
        Raises:
            Exception: if error occurs
        """        
        try:
            bestHyper = self.setBestHyper(best)
            hyperLayout = self.bestResult(score, bestHyper)
            #
            btn = QMessageBox.question(
                None,
                "Question",
                "Show the tuning grid?",
            )
            if btn == QMessageBox.Yes:
                gridLayout = self.tuneTable(means, params)
            else:
                layout = QVBoxLayout()
                self.replaceLayout.emit(layout)  
            #
            # need to convert from string to numeric
            self.hyper = self.hyperInstance.hyperToNumeric(bestHyper)      

        except Exception as e:
            print(f"tuneResults raised an exception: {e}")
            logger.error(f"tuneResults raised an exception: {str(e)}")
            raise Exception(f"Error in tuneResults: {str(e)}")

    def setBestHyper(self, subHyper: dict):
        """Apply tuning hyper parameters to copy of current hyper object

        Args:
            subHyper (dict): tuning parameters in numeric format

        Returns:
            dict: hyper parameters in string format
        """        
        # init best hyper
        bestHyper = copy.deepcopy(self.hyper)
        # convert from numeric to string
        # replace bestHyper with subHyper items
        for k, v in subHyper.items():
            if (k.startswith("model__")):
                kk = k.replace("model__", "")
                bestHyper[kk] = str(v)
            else:
                bestHyper[k] = str(v)
        #
        return bestHyper        


    def bestResult(self, bestScore: float, bestHyper: dict):
        """Display form of hyper parameters based on best tuning parameters

        Args:
            bestScore (float): Best score from tuning task
            bestHyper (dict): Best hyper parameters from tuning task
        """        
        self.hyperLayout = QVBoxLayout()
        #
        scoreForm = QFormLayout()
        #
        self.score = QLineEdit()
        self.score.setFixedWidth(600)
        self.score.setFont(self.font)
        self.score.setText(str(bestScore))
        scoreForm.addRow("Best Score:", self.score)
        #
        # convert from numeric to string
        hyperForm = self.hyperInstance.hyperForm(bestHyper)
        #
        self.scoreWij = QWidget() 
        self.scoreWij.setLayout(scoreForm)
        self.hyperWij = QWidget() 
        self.hyperWij.setLayout(hyperForm)
        #
        # add widgets to content
        self.hyperLayout.addWidget(self.scoreWij)
        self.hyperLayout.addSpacing(20)
        #
        # Create title
        title = QLabel("Best Grid:")
        title.setAlignment(Qt.AlignLeft) 
        title.setFont(self.font)
        self.hyperLayout.addWidget(title) 
        self.hyperLayout.addWidget(self.hyperWij)
        # add bottom spacing
        self.hyperLayout.addStretch(2)
        #
        central = QWidget()
        self.replaceWidget.emit(central)
        central.setLayout(self.hyperLayout)



    def tuneTable(self, means: list, params: list):
        """Create table to display grid search results

        Args:
            means (list): list of mean accuracy values
            params (list): list of dict for each set of grid search parameters
        Raises:
            Exception: if error occurs
        """        
        try:
            table = QTableWidget()
            layout = QVBoxLayout()
            # Create table title
            title = QLabel("Grid Search Results")
            title.setAlignment(Qt.AlignHCenter) 
            title.setStyleSheet("font-size: 20px; font-weight: bold;")
            layout.addWidget(title) 
            # do not use alignment on QTableWidget - the table will shrink! 
            layout.addWidget(table)      
            #
            # hide row numbering on left side of table
            table.verticalHeader().setVisible(False)
            #table.horizontalHeader().setVisible(False)
            #
            nrows = len(means)
            table.setRowCount(nrows)
            #
            # each item in params is a dict of 'tune param':'value' pairs
            # e.g. {'batch_size': 24, 'model__init_rate': 0.01, 'model__num_filters': 32, 'model__num_units': 100}
            #
            # use the first item in params to get the number of pairs
            # each param will be a separate col
            # add 1 for the mean accuracy column
            mcols = len(params[0]) + 1
            table.setColumnCount(mcols)
            #print("tuneTable nrows", nrows)
            #print("tuneTable mcols", mcols)
            #
            # The keys of the param:value pairs will be the table col headers.
            # Note: .keys() function returns a dict_keys, which looks like a list, but is not a list. It must be converted to a list as follows.
            colHdr = list(params[0].keys())
            colHdr.append("mean \naccuracy %")
            # remove the param prefix "model__"
            for col in range(mcols):
                table.setColumnWidth(col, 180)
                if(colHdr[col].startswith("model__")):
                    colHdr[col] = colHdr[col].replace("model__", "")
            #     
            #print("tuneTable after, colHdr", colHdr)   
            table.setHorizontalHeaderLabels(colHdr)
            #
            for row in range(nrows):
                mean = means[row]
                m = "{:.2f}".format(mean*100)
                # note:
                # .values() returns a dict_values view of param:value pairs
                # it must be converted to a list as follows.
                col_vals = list(params[row].values())
                # append mean value, to last column
                col_vals.append(m)
                #print("tuneTable col_vals", col_vals)
                for col in range(mcols):
                    # set each col value in current row
                    val = col_vals[col]
                    table.setItem(row, col, QTableWidgetItem(str(val)))
            hbox = QWidget()
            hlayout = QHBoxLayout(hbox)
            hlayout.setAlignment(Qt.AlignCenter) 
            okbtn = QPushButton("OK")
            okbtn.setFixedWidth(200)
            okbtn.setStyleSheet("background-color: #E0FFFF; border-radius: 5px; border-color: #DCDCDC; border-width: 2px; border-style: outset;")
            okbtn.setFont(self.font)
            hlayout.addWidget(okbtn)
            #
            layout.addWidget(hbox)
            #
            central = QWidget()
            self.replaceWidget.emit(central)
            central.setLayout(layout)
            #
            # connect okbtn button to function
            okbtn.clicked.connect(self.closeCentral)

        except Exception as e:
            print(f"tuneTable raised an exception: {e}")
            logger.error(f"tuneTable raised an exception: {str(e)}")
            raise Exception(f"Error in tuneTable: {str(e)}")

    def trainTaskFinished(self, include_val):
        """When the training task has complteted, plot the history.

        Args:
            include_val (bool): true if training included validation, false if training did not include validation
        """   
        # hist is keras.callbacks.History     
        hist = self.traintask.hist
        # need the history dictionary, hist.history
        history = hist.history
        # history contains two or four lists: accuracy, val_accuracy, loss, val_loss
        # epoch list derived from length of accuracy list
        # send signal to enable menu buttons
        self.trainingComplete.emit()
        #print("trainTaskFinished, hist type:", type(hist))
        #print("trainTaskFinished, history type:", type(history))
        #print("trainTaskFinished, history:", history)
        if(hist is not None):
            if(include_val):
                # history contains four lists: accuracy, val_accuracy, loss, val_loss
                ax = self.plotHistoryVal(history)
            else:
                # history contains two lists: accuracy, loss
                ax = self.plotHistory(history)    
        else:
            logger.error("Error in trainTaskFinished: hist is none")
            print("trainTaskFinished error: hist is none")

    def clearKeras(self):
        # the CPU thread is blocked and will not proceed until all pending GPU tasks on the current device are finished
        if(torch.get_default_device() == 'cuda'):
            torch.cuda.synchronize()   
            torch.cuda.empty_cache()
        # clear out old models and layers from the global state
        keras.backend.clear_session()
        

