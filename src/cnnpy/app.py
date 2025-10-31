#
import sys
import os
# disable worthless tensorflow messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# disable scaling on Windows 11
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
# can help prevent fragmentation on GPU
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"
from pathlib import Path
# for splash screen
import resources_rc
import logging
logger = logging.getLogger()
#
import cProfile 
from pstats import SortKey, Stats
#
import PySide6
from PySide6.QtWidgets import (
    QApplication,
    QSplashScreen,
    QMainWindow,
    QVBoxLayout,
    QLayout,
    QMenu,
    QStatusBar,
    QWidget
)
from PySide6.QtGui import QFont, QPixmap, QAction
from PySide6.QtCore import Qt, Signal, Slot, QTimer
#
from appctrl.about_dialog import AboutDialog
from appctrl.tutor_dialog import TutorDialog
from appctrl.controller import Controller
from appctrl.app_util import Util
from theprep.data_params import DataParams
from theprep.hyper_params import HyperParams
from thenet.netmodel import NetModel
#
import torch
os.environ["KERAS_BACKEND"] = "torch"

#
# Subclass QMainWindow to customize main window
class AppWin(QMainWindow):

    # Class variables
    # called like this: AppWin.font
    font = QFont("Arial", 14, QFont.Weight.Bold)
    # need QWIDGETSIZE_MAX to allow window resizing
    QWIDGETSIZE_MAX = 16777215
    #
    #
    def __init__(self):
        """Initialize the application

        Set window size, and allow rezing.
        Set GPU if available
        Initialize the file logger
        Create various signal connections
        Create menus
        Initialize central window widget

        Note: some methods are decorated with @Slot(), meaning they can
        receive signals. However, the decorators are completely optional. The sender and receiver methods must be 'connected' for signals to actually work.
        """        
        super().__init__()
        #
        # QTimer did not help to show splash screen
        #QTimer.singleShot(100, self.initGui)
        #
        self.initGui()
        print("CNN App ready")

    def initGui(self):
        #
        # set size and font
        self.initWin()
        #
        # set torch with gpu or cpu
        self.initTorch()
        #
        # set file logger
        # don't log during startup, since writing to file may 
        # slow down startup process
        self.initLogger()
        #
        # setup signal connections
        self.connections()
        #
        # create menu bar, status bar
        self.createBars()
        #
        # initialize central widget
        self.createCentral()

    def initWin(self):
        #print("AppWin initializing")
        title = "CNN App"
        self.setFont(AppWin.font)
        self.setWindowTitle(title)
        w = 1200
        h = 900
        # initial window size in px
        self.setFixedSize(w, h)
        # To allow resizing
        self.setFixedSize(AppWin.QWIDGETSIZE_MAX, AppWin.QWIDGETSIZE_MAX) 
        #
        #version = Util.appVersion()
        #print("app version:", version)        


    def initTorch(self):
        # set gpu if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            #print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            #print("No GPU available, using CPU.")
        #
        #print("torch.device: ", device)
        torch.set_default_device(device)


    def initLogger(self):
        """Initialize logger with file handler
        """        
        #print("App initLogger")
        handler=logging.FileHandler(filename='myapp.log', mode='w')
        #handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # to record to logger, must include logger.setLevel
        logger.setLevel(logging.INFO)
        #logger.info('Started')


    def createBars(self):
        """Create menu bar and status bar

        Also, create app menu, task menu, and help menu
        """        
        #print("App createBasic")
        self.createMenuBar()
        #  self.createToolBar()
        self.createStatusBar()
        #
        self.createAppMenu()
        self.createTaskMenu()
        self.creatHelpMenu()

    def createMenuBar(self):
        #print("App createMenu")
        self.mbar = self.menuBar()
        #self.mbar.setFont(AppWin.font)
        #self.mbar.setNativeMenuBar(False)

    def createStatusBar(self):
        #print("App createStatusBar")
        self.status = QStatusBar()
        self.status.showMessage("Status here")
        self.setStatusBar(self.status)


    def connections(self):
        """
        Create signal connections to AppWin 'receiver' methods from 'sender' methods in the following classes:

        Controller: the middleware between the GUI and the backend modules. 
        Note: Another approach would be to set the AppWin instance inside the 
        Controller, just like self.netcontrol.setDataParams and self.netcontrol.setHyperParams. Then, instead of signal connections, the AppWin methods could be called directly within Controller. Either way, many of these method calls are made to enable menu items that are originally disabled to prevent the user from jumping out of sequence.

        DataParams: class for parameters related to the datasets

        HyperParams: class for hyper parameters, which are used to specify the neural network model

        NetModel: class for creating a new network model, or saving and loading a trained network model

        Note: <sender> connect to <receiver>
        """
        #print("App connections")
        self.netcontrol = Controller()
        # signal sender connect to slot receiver
        self.netcontrol.trainingDataIsLoaded.connect(self.initNetwork) 
        self.netcontrol.trainingDataIsLoaded.connect(self.updateTuning) 
        # signal to allow tasks to run, after network model is initialized
        self.netcontrol.prepComplete.connect(self.updateTasks)
        self.netcontrol.trainingComplete.connect(self.updateModelAct)  
        self.netcontrol.testingDataIsLoaded.connect(self.updateTesting) 
        self.netcontrol.updateStatus.connect(self.setStatus)  
        self.netcontrol.replaceLayout.connect(self.replaceCentralLayout) 
        self.netcontrol.replaceWidget.connect(self.replaceCentralWidget) 
        #
        self.dataParams = DataParams() 
        self.netcontrol.setDataParams(self.dataParams)
        # after data params has been loaded, allow hyper params to be loaded 
        self.dataParams.dataIsSet.connect(self.updateHyper) 
        self.dataParams.dataIsSet.connect(self.netcontrol.initModelDir) 
        self.dataParams.updateStatus.connect(self.setStatus)  
        self.dataParams.replaceLayout.connect(self.replaceCentralLayout) 
        self.dataParams.replaceWidget.connect(self.replaceCentralWidget) 
        #
        self.hyperParams = HyperParams() 
        self.netcontrol.setHyperParams(self.hyperParams) 
        # after hyper params has been loaded, allow training datasets to be loaded 
        self.hyperParams.hyperIsSet.connect(self.updatePostHyper)
        self.hyperParams.updateStatus.connect(self.setStatus) 
        self.hyperParams.replaceLayout.connect(self.replaceCentralLayout)
        self.hyperParams.replaceWidget.connect(self.replaceCentralWidget) 
        # 
        self.netModel = NetModel()  
        self.netcontrol.setNetModel(self.netModel)
        self.netModel.updateStatus.connect(self.setStatus)
        self.netModel.loadComplete.connect(self.updateModelAct)  

    def getController(self):
        return self.netcontrol

    def createAppMenu(self):
        #print("App createAppMenu")
        #logger.info('App createAppMenu')
        appMenu = QMenu("App", self)
        self.mbar.addMenu(appMenu)
        appMenu.setFont(AppWin.font)
        #
        # only separator style works
        appMenu.setStyleSheet("""
            QMenu::separator 
            { background-color: #B6B6B4; height: 0.1em; }
            """)
        #
        newParamsAct = QAction("New Data Params", self)
        appMenu.addAction(newParamsAct)
        newParamsAct.triggered.connect(self.dataParams.newParams)
        #
        loadParamsAct = QAction("Load Data Params", self)
        appMenu.addAction(loadParamsAct)
        loadParamsAct.triggered.connect(self.dataParams.loadParams)
        self.params_loaded = False
        #
        editParamsAct = QAction("Edit Data Params", self)
        appMenu.addAction(editParamsAct)
        editParamsAct.triggered.connect(self.dataParams.editParams)
        #
        saveParamsAct = QAction("Save Data Params", self)
        appMenu.addAction(saveParamsAct)
        saveParamsAct.triggered.connect(self.dataParams.saveParams)
        appMenu.addSeparator()
        #
        # note: editHyperAct is under Tasks
        newHyperAct = QAction("New Hyper Params", self)
        appMenu.addAction(newHyperAct)
        # note: editHyperAct is under Tasks
        newHyperAct.triggered.connect(self.hyperParams.newParams)
        #
        self.loadHyperAct = QAction("Load Hyper Params", self)
        appMenu.addAction(self.loadHyperAct)
        self.loadHyperAct.triggered.connect(self.hyperParams.loadParams)
        self.loadHyperAct.setEnabled(False)
        self.hyper_loaded = False
        #
        saveHyperAct = QAction("Save Hyper Params", self)
        appMenu.addAction(saveHyperAct)
        saveHyperAct.triggered.connect(self.hyperParams.saveParams)
        appMenu.addSeparator()
        #
        self.loadTrainDataAct = QAction("Load Training Dataset", self)
        appMenu.addAction(self.loadTrainDataAct)
        self.loadTrainDataAct.triggered.connect(self.netcontrol.prepTrainingData)
        self.loadTrainDataAct.setEnabled(False)
        self.train_loaded = False

        #
        loadModelAct = QAction("Load Network Model", self)
        appMenu.addAction(loadModelAct)
        self.model_loaded = False
        # load a neural network model
        loadModelAct.triggered.connect(self.netModel.loadModel)
        #
        self.saveModelAct = QAction("Save Network Model", self)
        appMenu.addAction(self.saveModelAct)
        # save the neural network model
        self.saveModelAct.triggered.connect(self.netModel.saveModel)
        self.saveModelAct.setEnabled(False)

        #
        self.loadTestDataAct = QAction("Load Test Dataset", self)
        appMenu.addAction(self.loadTestDataAct)
        self.loadTestDataAct.triggered.connect(self.netcontrol.prepTestingData)
        self.loadTestDataAct.setEnabled(False)
        #
        appMenu.addSeparator()
        #
        appMenu.addAction("&Exit", self.close)

    def createTaskMenu(self):
        #print("App createTaskMenu")
        #logger.info('App createTaskMenu')
        tasks = QMenu("Tasks", self)
        self.mbar.addMenu(tasks)
        tasks.setFont(AppWin.font)
        #
        self.editHyperAct = QAction("Edit Hyper Params", self)
        tasks.addAction(self.editHyperAct)
        self.editHyperAct.triggered.connect(self.hyperParams.editParams)
        self.editHyperAct.setEnabled(False)
        #
        self.tuningAct = QAction("Tuning", self) 
        tasks.addAction(self.tuningAct)
        self.tuningAct.triggered.connect(self.netcontrol.runTuner)
        self.tuningAct.setEnabled(False)
        #
        self.tuningStopAct = QAction("Stop Tuning Task", self) 
        tasks.addAction(self.tuningStopAct)
        self.tuningStopAct.triggered.connect(self.netcontrol.stopTuner)
        self.tuningStopAct.setEnabled(False)

        #
        self.initNetworkAct = QAction("Initialize New Network Model", self)   
        tasks.addAction(self.initNetworkAct)
        self.initNetworkAct.triggered.connect(self.netcontrol.trainPrep)
        self.initNetworkAct.setEnabled(False)
        #
        self.trainingAct = QAction("Training Only", self)
        tasks.addAction(self.trainingAct)
        self.trainingAct.setEnabled(False)
        # using lambda as no arg function for connect
        # which allows the actual function to pass an argument
        self.trainingAct.triggered.connect(lambda: self.netcontrol.train2(include_val=False))
        #
        self.trainValAct = QAction("Training with Validation", self)
        tasks.addAction(self.trainValAct)
        self.trainValAct.setEnabled(False)
        self.trainValAct.triggered.connect(lambda: self.netcontrol.train2(include_val=True))
        #
        self.evaluationAct = QAction("Evaluation", self)
        tasks.addAction(self.evaluationAct)
        self.evaluationAct.setEnabled(False)
        self.evaluationAct.triggered.connect(self.netcontrol.evaluate)
        #        
        self.predictionAct = QAction("Prediction", self)
        tasks.addAction(self.predictionAct)
        self.predictionAct.setEnabled(False)
        self.predictionAct.triggered.connect(self.netcontrol.predict)
        #

    def creatHelpMenu(self):  
        #print("App creatHelpMenu")  
        help = QMenu("Help", self)
        self.mbar.addMenu(help)
        help.setFont(AppWin.font)
        #
        self.snapAct = QAction("Snap", self)
        help.addAction(self.snapAct)
        self.snapAct.triggered.connect(self.saveWidget)
        #
        self.tutorAct = QAction("Tutorial", self)
        help.addAction(self.tutorAct)
        self.tutorAct.triggered.connect(self.showTutor)
        #
        self.aboutAct = QAction("About", self)
        self.aboutAct.triggered.connect(self.showAbout)
        help.addAction(self.aboutAct)

    def showTutor(self):
        #logger.info('AppWin, tutorial')
        tutor = TutorDialog(self)
        # show() for non-modal dialog
        tutor.show()

    def showAbout(self):
        #logger.info('AppWin, about')
        dialog = AboutDialog()
        # exec() for modal dialog
        dialog.exec()

    @Slot()
    def setStatus(self, msg: str):
        self.status.showMessage(msg) 

    # Define a slot to handle the custom signal
    @Slot()
    def updateHyper(self):
        self.params_loaded = True
        # enable loadHyperAct when data params are set
        self.loadHyperAct.setEnabled(True)  
        if(self.model_loaded):  
            self.loadTestDataAct.setEnabled(True)

    # Define a slot to handle the custom signal
    @Slot()
    def updatePostHyper(self):
        self.hyper_loaded = True
        # enable loadTrainDataAct when hyper params are set
        self.loadTrainDataAct.setEnabled(True)
        # enable editHyperAct when hyper params are set
        self.editHyperAct.setEnabled(True)    

    # Define a slot to handle the custom signal
    @Slot()
    def initNetwork(self):
        # enable task QActions when training dataset has been loaded
        self.train_loaded = True
        self.initNetworkAct.setEnabled(True)

    # Define a slot to handle the custom signal
    @Slot()
    def updateTuning(self):
        # enable task QActions when network model is initialized
        if(self.train_loaded and self.hyper_loaded):
            self.tuningAct.setEnabled(True)
            self.tuningStopAct.setEnabled(True)

    # Define a slot to handle the custom signal
    @Slot()
    def updateTasks(self):
        # enable task QActions when network model is initialized
        if(self.train_loaded):
            self.trainingAct.setEnabled(True)
            self.trainValAct.setEnabled(True)
        if(self.params_loaded):    
            self.loadTestDataAct.setEnabled(True) 

    # Define a slot to handle the custom signal
    @Slot()
    def updateModelAct(self):
        # enable task QActions when training is completed or model is loaded
        self.model_loaded = True
        self.saveModelAct.setEnabled(True)
        # if model is loaded, disable training
        self.trainingAct.setEnabled(False)
        self.trainValAct.setEnabled(False)
        if(self.params_loaded):    
            self.loadTestDataAct.setEnabled(True) 

    # Define a slot to handle the custom signal
    @Slot()
    def updateTesting(self):
        # enable task QActions when testing dataset is loaded
        self.evaluationAct.setEnabled(True)
        self.predictionAct.setEnabled(True)

    def createCentral(self):
        """Create empty central widget
        """        
        widget = QWidget()
        self.replaceCentral(widget)

    # Define a slot to handle the custom signal
    @Slot(QWidget)
    def replaceCentralWidget(self, item: QWidget):
        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(item)
        # Central Widget
        widget = QWidget()
        widget.setLayout(layout)
        # replace CentralWidget
        self.replaceCentral(widget)  

    # Define a slot to handle the custom signal
    @Slot(QLayout)
    def replaceCentralLayout(self, layout: QLayout):
        # Central Widget
        widget = QWidget()
        widget.setLayout(layout)
        # replace CentralWidget
        self.replaceCentral(widget)  

    def replaceCentral(self, widget: QWidget):
        # first clear old central
        if(self.centralWidget is not None):
            if(isinstance(self.centralWidget, QWidget)):
                # clear children from old central
                Util.clearChildren(self.centralWidget)
        # replace CentralWidget
        self.centralWidget = widget
        self.setCentralWidget(widget) 
        # clear status 
        self.status.clearMessage()

    def saveWidget(self):
        try:
            # Grab  central widget
            if not self.centralWidget.children():
                Util.alert("Widget is empty.")
                return
            # get window image    
            pixmap = self.centralWidget.grab() 
            if(pixmap.isNull()):
                Util.alert("Widget is not set.")
                return
            #
            result_dir = self.netcontrol.getResultDir()
            if(result_dir is not None):
                full = Util.saveFileDialog(title="Save Window as .png:", filters=["PNG files (*.png);;All Files (*)"], dir=result_dir)
                print("saveWidget, full: ", full)
                if full:
                    # Save the QPixmap to a file
                    if pixmap.save(str(full), "PNG"):
                        print(f"widget saved to {full}")
                    else:
                        print("Failed to save widget.")
                else:
                    # user canceled dialog without selecting a path
                    logger.error("No path selected, widget not saved")

        except Exception as e:
            logger.error(f"Error in saveWidget: {str(e)}")
            print(f"Error in saveWidget: {e}")
            raise Exception("Error in saveWidget") 

def initSplash(app):
    pixmap = QPixmap(":/icons/logo.png")
    if pixmap.isNull():
        splash = None
        print("Splash screen image not loaded.")
    else:    
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()
        splash.showMessage("CNN App loading modules...", Qt.AlignTop | Qt.AlignHCenter)

    # need modal dialog to allow splash screen to load ??
    Util.autoCloseMessageBox(300, "Loading")
    return splash


# Note, main is defined outside of class
def main():
    #print("app main starting")
    #print(sys.path)
    # app with no command line args
    app = QApplication([])
    #
    # need splash, too slow ??
    # app opens quickly without much printing or logging
    #splash = initSplash(app)
    #
    # AppWin: subclass of QMainWindow for menu, toolbar, status
    win = AppWin()  
    win.show()
    #
    #if(splash is not None):
    #    splash.finish(win)
    
    # begin event loop
    app.exec()

if __name__ == "__main__":
    main()
   #cProfile.run('main()')