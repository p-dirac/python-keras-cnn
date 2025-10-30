import os
from pathlib import Path
import logging
logger = logging.getLogger()
import json
from PySide6.QtWidgets import (
    QPushButton,
    QLayout,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QLabel,
    QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from appctrl.app_util import Util
from PySide6.QtCore import Signal, QObject
#
class DataParams(QObject):

    """
    Params data for convolution network.

    """
    # Class variables
    # send signal when data params has been set
    # signal connections must be set in app_main
    dataIsSet = Signal(bool)
    updateStatus = Signal(str)
    replaceLayout = Signal(QLayout)
    replaceWidget = Signal(QWidget)    
    #
    # app params folder, which stores data param files and hyper param files
    APP_INPUT_SUBDIR = "data/app_input"
    
    # constructor function
    def __init__(self):
        # must include super call for QObject, Signal
        super().__init__()
        #
        # data_file: base name of data params file
        self.data_file = None
        #
        # params: actual contents of data_file
        self.params = None
        #
        # default values for params; will be reset from prepForm
        self.num_classes = 1
        self.train_dir = "../datasets/<data>/training"
        self.test_dir = "../datasets/<data>/testing"
        # note: square image sizes simplify input and computations 
        # image_size: width, height of each image in input dataset
        self.image_size = 2
        self.result_dir = "../../Results/<cnn>"
        #
        self.font = QFont("Arial", 14, QFont.Weight.Bold)
        self.default_params = {
            "num_classes": 1,
            "train_dir": "C:\\Datasets\\<my_data>\\training",
            "test_dir": "C:\\Datasets\\<my_data>\\testing",
            "image_size": 2,
            "result_dir": "C:\\Results\\<cnn>"
        }

    def getParams(self):
        return self.params
        
    def editParams(self):
        p = self.params
        if (p is None) or (not p) :
            # set default params, which user may edit
            p = self.default_params
        self.paramsForm(p)

    def paramsForm(self, p: dict):
        #print("paramsForm: ", p)
        if (p is None) or (not p) :
            # set default params, which user may edit
            p = self.default_params
        central = QWidget()
        self.replaceWidget.emit(central)
        vlayout = QVBoxLayout()
        central.setLayout(vlayout)
        # Create form title
        title = QLabel("Data Parameters")
        title.setAlignment(Qt.AlignHCenter) 
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        vlayout.addWidget(title) 
        #
        prepLayout = self.prepForm(p)
        prep = QWidget()
        prep.setLayout(prepLayout)
        vlayout.addWidget(prep)
        #
        hbox = QWidget()
        hlayout = QHBoxLayout(hbox)
        hlayout.setAlignment(Qt.AlignHCenter) 
        #
        #
        saveAll = QPushButton("OK")
        saveAll.setFixedWidth(200)
        saveAll.setStyleSheet("background-color: #E0FFFF; border-radius: 5px; border-color: #DCDCDC; border-width: 2px; border-style: outset;")
        saveAll.setFont(self.font)
        hlayout.addWidget(saveAll)
        #
        # connect saveAll button to function
        saveAll.clicked.connect(self.updateParams)
        #
        # must call spacing in order; this spacing is between
        # the two buttons
        hlayout.addSpacing(70)
        #
        cancel = QPushButton("Cancel")
        cancel.setFixedWidth(200)
        cancel.setStyleSheet("background-color: #FFFFE0;border-radius: 5px; border-color: #DCDCDC; border-width: 2px; border-style: outset;")
        cancel.setFont(self.font)
        hlayout.addWidget(cancel)
        #
        cancel.clicked.connect(self.closeCentral)
        #
        vlayout.addWidget(hbox)
        vlayout.addStretch(0) 

    def prepForm(self, prep: dict):
        #print("prepForm: ", prep)
        layout = QFormLayout()
        #layout.setFormAlignment(Qt.AlignCenter)
        # Add widgets to the layout
        self.num_classes = prep.get("num_classes")
        tr = prep.get("train_dir")
        te = prep.get("test_dir")
        sz = prep.get("image_size")
        rs = prep.get("result_dir")
        #print("num_classes: ", self.num_classes)
        #print("train_dir: ", tr)
        #print("test_dir: ", te)
        #print("image_size: ", sz)
        #
        self.prep1 = QLineEdit()
        self.prep1.setFixedWidth(600)
        self.prep1.setFont(self.font)
        self.prep1.setText(str(self.num_classes))
        # to set font, use QLabel
        #prep1Name = QLabel("Number of classes:")
        #prep1Name.setFont(self.font)
        #layout.addRow(prep1Name, self.prep1)
        layout.addRow("Number of classes:", self.prep1)
        #
        self.prep2 = QLineEdit()
        self.prep2.setFixedWidth(600)
        self.prep2.setFont(self.font)
        self.prep2.setText(tr)
        layout.addRow("Training directory:", self.prep2)
        #
        self.prep3 = QLineEdit()
        self.prep3.setFixedWidth(600)
        self.prep3.setFont(self.font)
        self.prep3.setText(te)
        layout.addRow("Testing directory:", self.prep3)
        #
        self.prep4 = QLineEdit()
        self.prep4.setFixedWidth(600)
        self.prep4.setFont(self.font)
        self.prep4.setText(str(sz))
        layout.addRow("Image size:", self.prep4)
        #
        self.prep5 = QLineEdit()
        self.prep5.setFixedWidth(600)
        self.prep5.setFont(self.font)
        self.prep5.setText(rs)
        layout.addRow("Results directory:", self.prep5)
        #
        return layout

    def updateParams(self):
        # save params here for output file
        self.params = self.savePrep()
        #
        #print("updateParams, params: ", self.params)
        #
        # send signal to enable menu buttons
        self.dataIsSet.emit(True)
        #
        self.closeCentral()
        self.updateStatus.emit("Params updated") 

    def savePrep(self):
        # from prepForm
        num_classes = int(self.prep1.text())
        train_dir = self.prep2.text()
        test_dir = self.prep3.text()
        image_size = int(self.prep4.text())
        result_dir = self.prep5.text()
        prep = {
            "num_classes" : num_classes,
            "train_dir" : train_dir,
            "test_dir" : test_dir,
            "image_size" : image_size,
            "result_dir": result_dir
        }
        #print("savePrep, prep: ", prep)
        return prep

    def closeCentral(self):
        central = QWidget()
        self.replaceWidget.emit(central)

    def newParams(self):
        p = None
        # will use default params
        self.paramsForm(p)

    def loadParams(self):
        p = self.readParams() 
        if(bool(p)):
            self.params = p
            print("data params loaded: ", p)
            logger.info(f"data params loaded: {p}")
            self.paramsForm(p) 
            self.updateStatus.emit("Data Params loaded") 

    def readParams(self):
        """Read parameters from file

        Args:
            None

        Raises:
            Exception: if error occurs

        Returns:
            dict : network parameters 
        """
        try:
            full_dir = os.path.join(Path.cwd(), DataParams.APP_INPUT_SUBDIR)
            file_path = Util.openFileDialog(title="Open JSON data params file", filters=["JSON files (*.json);;All Files (*)"], dir=full_dir)
            # Read from file and parse JSON
            print("readParams, file_path: ", file_path)
            if file_path:
                # open file for reading
                with open(file_path, "r") as f:
                    # read params from json file and deserialize to dict format
                    pj = json.load(f)
                p = dict(pj)    
                self.data_file = os.path.basename(file_path)
            else:
                # file does not exist
                # make copy of default params ?
                #p = dict(self.params)  
                p = dict()     
        except Exception:
            logger.error("Error in readParams")
            raise Exception("Error in readParams") 
            Util.alert("Error in readParams.")
        else:    
            return p

    def saveParams(self):
        if (self.params is None) or (not self.params) :
            Util.alert("Data Params is not set.")
            return
        else:    
            # save params 
            stat = self.saveParamsCwd()
            return
        
    def saveParamsCwd(self):
        """Write parameters to current directory

        Raises:
            Exception: if error occurs

        """
        try:
            #print("saveParamsCwd, params: ", self.params)
            #print("saveParamsCwd, data_file: ", self.data_file)
            full_dir = os.path.join(Path.cwd(), DataParams.APP_INPUT_SUBDIR)
            full = Util.saveFileDialog(title="Save As .json:", filters=["JSON files (*.json);;All Files (*)"], dir=full_dir)
            print("saveParams, full: ", full)
            if full:
                # open file for writing
                with open(full, "w") as f:
                    #write params dict to file in json format
                    json.dump(self.params, f, indent=4)
                self.updateStatus.emit("Data Params saved to file") 
                logger.info(f"Data Params saved to file: {full}")
            else:
                # user canceled dialog without selecting a path
                logger.error("No path selected, params not saved")
        except Exception:
            logger.error("Error in saveParamsCwd")
            Util.alert("DataParams: Error in saveParamsCwd.")
            raise Exception("Error in saveParamsCwd")
            
