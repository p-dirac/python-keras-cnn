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
class HyperParams(QObject):

    """
    Hyper Parameters for convolution network.

    """
    # Class variables
    # send signal when hyper has been set
    # signal connections must be set in app_main
    hyperIsSet = Signal(bool)
    updateStatus = Signal(str)
    replaceLayout = Signal(QLayout)
    replaceWidget = Signal(QWidget)

    #
    # app params folder, which stores hyper param files
    APP_INPUT_SUBDIR = "data/app_input"
    
    # constructor function
    def __init__(self):
        # must include super call for QObject, Signal
        super().__init__()
        #
        # hyper_file: base name of hyper params file
        self.hyper_file = None
        #
        # hyper: actual contents of hyper_file
        self.hyper = None
        #
        # defaultHyper: default values for hyper
        # Note: list values are used in the grid search tuning process.
        # Note: in the training process, only the first value of each parameter is used.
        self.defaultHyper = {
            "init_rate": [0.02,0.01],
            "decay_steps": 400,
            "decay_rate": 0.9,
            "kernel_size": 3,
            "num_filters": [32,40],
            "num_units": [512,800],
            "momentum": 0.9,
            "drop_out": 0.2,
            "epochs": [5,10],
            "batch_size": [64,80]
        }
        self.font = QFont("Arial", 14, QFont.Weight.Bold)

    def getHyper(self):

        return self.hyper
        
    def editParams(self):
        p = self.hyper
        if (p is None) or (not p) :
            # set default params, which user may edit
            p = self.defaultHyper
        self.paramsForm(p)

    def paramsForm(self, p: dict):
        #print("hyper paramsForm: ", p)
        if (p is None) or (not p) :
            # set default params, which user may edit
            p = self.defaultHyper
        #
        central = QWidget()
        self.replaceWidget.emit(central)
        vlayout = QVBoxLayout()
        central.setLayout(vlayout)
        #
        # Create form title
        title = QLabel("Hyper Parameters \n (csv entries accept multiple values for tuning)")
        title.setAlignment(Qt.AlignHCenter) 
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        vlayout.addWidget(title) 
        #
        hyperLayout = self.hyperForm(p)
        hbox = QWidget()
        hbox.setLayout(hyperLayout)
        vlayout.addWidget(hbox)
        #
        hbox = QWidget()
        hlayout = QHBoxLayout(hbox)
        hlayout.setAlignment(Qt.AlignHCenter) 
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

    def hyperForm(self, hyper:dict):
        #print("hyperForm: ", hyper)
        layout = QFormLayout()
        #layout.setFormAlignment(Qt.AlignCenter)
        # Add widgets to the layout
        rate = hyper["init_rate"]
        steps = hyper["decay_steps"]
        decay = hyper["decay_rate"]
        #
        self.sch1 = QLineEdit()
        self.sch1.setFixedWidth(600)
        self.sch1.setFont(self.font)
        # if rate is list of foats, convert each to str, then join to csv
        self.sch1.setText(self.toCSV(rate))
        layout.addRow("Initial rate (csv):", self.sch1)
        #
        self.sch2 = QLineEdit()
        self.sch2.setFixedWidth(600)
        self.sch2.setFont(self.font)
        self.sch2.setText(str(steps))
        layout.addRow("Decay steps:", self.sch2)
        #
        self.sch3 = QLineEdit()
        self.sch3.setFixedWidth(600)
        self.sch3.setFont(self.font)
        self.sch3.setText(str(decay))
        layout.addRow("Decay rate:", self.sch3)
        #
        kernel_size = hyper["kernel_size"]
        num_filters = hyper["num_filters"]
        num_units = hyper["num_units"]
        #
        self.ed0 = QLineEdit()
        self.ed0.setFixedWidth(600)
        self.ed0.setFont(self.font)
        # convert list of num_filters to csv str 
        self.ed0.setText(str(kernel_size))
        layout.addRow("Kernal size:", self.ed0)

        self.ed1 = QLineEdit()
        self.ed1.setFixedWidth(600)
        self.ed1.setFont(self.font)
        # convert list of num_filters to csv str 
        self.ed1.setText(self.toCSV(num_filters))
        layout.addRow("Num filters (csv):", self.ed1)
        #
        self.ed2 = QLineEdit()
        self.ed2.setFixedWidth(600)
        self.ed2.setFont(self.font)
        # convert list of num_units to csv str 
        self.ed2.setText(self.toCSV(num_units))
        layout.addRow("Num units (csv):", self.ed2)
        #
        momentum = hyper["momentum"]
        drop_out = hyper["drop_out"]
        #
        self.ed3 = QLineEdit()
        self.ed3.setFixedWidth(600)
        self.ed3.setFont(self.font)
        self.ed3.setText(str(momentum))
        layout.addRow("Momentum:", self.ed3)
        #
        self.ed4 = QLineEdit()
        self.ed4.setFixedWidth(600)
        self.ed4.setFont(self.font)
        self.ed4.setText(str(drop_out))
        layout.addRow("Drop out:", self.ed4)
        #
        epochs = hyper["epochs"]
        batch = hyper["batch_size"]
        #
        self.trn1 = QLineEdit()
        self.trn1.setFixedWidth(600)
        self.trn1.setFont(self.font)
        # convert list of epochs to csv str 
        self.trn1.setText(self.toCSV(epochs))
        layout.addRow("Epochs (csv):", self.trn1)
        #
        self.trn2 = QLineEdit()
        self.trn2.setFixedWidth(600)
        self.trn1.setFont(self.font)
        # convert list of batch sizes to csv str 
        self.trn2.setText(self.toCSV(batch))
        layout.addRow("Batch size (csv):", self.trn2)
        #
        return layout


    def updateParams(self):
        # save hyper here for output file
        # convert str to numeric
        self.hyper = self.updateHyper() 
        #print("updateParams, hyper: ", self.hyper)
        #
        #logger.info('updateParams, calling emit')
        # send signal to enable menu buttons
        self.hyperIsSet.emit(True)
        #
        self.closeCentral()
        self.updateStatus.emit("Hyper Params updated") 

        
    def updateHyper(self):
        """From form layout, convert text to numeric values

        Returns:
            dict: hyper parameters with numeric values
        """        
        # from learnForm
        # self.sch1.text() is in csv format
        # split creates list of items, map converts each item to float
        # init_rate is list of floats
        init_rate = self.csvToFloat(self.sch1.text()) 
        decay_steps = int(self.sch2.text())
        decay_rate = float(self.sch3.text())
        kernel_size = int(self.ed0.text())
        #num_filters = int(self.ed1.text())
        # num_filters is list of ints, converted from csv string
        num_filters = self.csvToInt(self.ed1.text())  
        #num_units = int(self.ed2.text())
        # num_units is list of ints
        num_units = self.csvToInt(self.ed2.text())
        momentum = float(self.ed3.text())
        drop_out = float(self.ed4.text())
        #epochs = int(self.trn1.text())
        # epochs is list of ints
        epochs = self.csvToInt(self.trn1.text())
        #batch_size = int(self.trn2.text())
        # batch_size is list of ints
        batch_size = self.csvToInt(self.trn2.text()) 
        #
        # when using init_rate, check with: if isinstance(init_rate, list): 
        hyper = {
        "init_rate" : init_rate,
        "decay_steps" : decay_steps,
        "decay_rate" : decay_rate,
         "kernel_size" : kernel_size,
        "num_filters" : num_filters,
        "num_units" : num_units,
        "momentum" : momentum,
        "drop_out" : drop_out,
        "epochs" : epochs,
        "batch_size" : batch_size
        }
        return hyper

    def hyperToNumeric(self, hype: dict):
        """From string format, convert text to numeric values

        Update numeric hyper parameters

        Args:
           hype (str): hyper parameters in string format
        """        
        self.hyper = {
        "init_rate" : self.csvToFloat(hype["init_rate"]),
        "decay_steps" : int(hype["decay_steps"]),
        "decay_rate" : float(hype["decay_rate"]),
         "kernel_size" : int(hype["kernel_size"]),
        "num_filters" : self.csvToInt(hype["num_filters"]) ,
        "num_units" : self.csvToInt(hype["num_units"]),
        "momentum" : float(hype["momentum"]),
        "drop_out" : float(hype["drop_out"]),
        "epochs" : self.csvToInt(hype["epochs"]),
        "batch_size" : self.csvToInt(hype["batch_size"]) 
        }

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
            print("hyper params loaded: ", p)
            logger.info(f"hyper params loaded: {p}")
            self.paramsForm(p) 
            self.updateStatus.emit("Hyper Params loaded") 

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
            full_dir = os.path.join(Path.cwd(), HyperParams.APP_INPUT_SUBDIR)
            file_path = Util.openFileDialog(title="Open JSON hyper params file", filters=["JSON files (*.json);;All Files (*)"], dir=full_dir)
            # Read from file and parse JSON
            print("readParams, file_path: ", file_path)
            if file_path:
                # open file for reading
                with open(file_path, "r") as f:
                    # read params from json file and deserialize to dict format
                    pj = json.load(f)
                # overwrite default params    
                p = dict(pj)
                self.hyper_file = os.path.basename(file_path)
            else:
                # file does not exist
                # make copy of default params ?
                p = dict()     
        except Exception:
            logger.error("Error in readParams")
            raise Exception("Error in readParams") 
            Util.alert("Error in readParams.")
        else:    
            return p

    def saveParams(self):
        if (self.params is None) or (not self.params) :
            Util.alert("Hyper Params is not set.")
            return
        else:    
            # save params 
            stat = self.saveHyper()
            return


    def saveHyper(self):
        """Write hyper parameters to app params directory

        If file already exists, user will get confirmation dialog to overwrite file.

        Raises:
            Exception: if error occurs

        Returns:
            int: -1 if issue, else 0
        """
        try:
            #print("saveHyper, hyper: ", self.hyper)
            #print("saveHyper, hyper_file: ", self.hyper_file)
            full_dir = os.path.join(Path.cwd(), HyperParams.APP_INPUT_SUBDIR)
            full = Util.saveFileDialog(title="Save As .json:", filters=["JSON files (*.json);;All Files (*)"], dir=full_dir)
            print("saveHyper, full: ", full)
            if full:
                # open file for writing
                with open(full, "w") as f:
                    #write params dict to file in json format
                    json.dump(self.hyper, f, indent=4)
                self.updateStatus.emit("Hyper Params saved to file") 
                logger.info(f"Hyper Params saved to file: {full}")
            else:
                # user canceled dialog without selecting a path
                logger.info("No path selected, hyper params not saved")
        except Exception:
            logger.error("Error in saveHyper")
            raise Exception("Error in saveHyper")
            Util.alert("Error in saveHyper.")

    def toCSV(self, val):

        """Convert given val to csv string

        The type of val could be str, int, float, or a list of one type.

        If val is already a list, map each list item to a str, then join to create a csv string.

        If val is not a list, make it a list, map each list item to a str, then join to create a csv string.

        Returns:
            str: csv string
        """  
        # assume val is already a list      
        v = val
        if not isinstance(val, list):
            # val not a list, make it a list
            v = [val]
        # map each list item to a str, then join to create a csv string    
        csv = ','.join(map(str, v))    
        return csv  

    def csvToFloat(self, csv):
        """Convert csv string into list of floats

        Args:
            csv (str): csv string to be converted

        Returns:
            list: list of floats
        """        

        # split csv string into list, map each list item to float,
        # then create list of floats
        float_list = list( map(float, csv.split(",")) )   
        return float_list     

    def csvToInt(self, csv):
        """Convert csv string into list of ints

        Args:
            csv (str): csv string to be converted

        Returns:
            list: list of ints
        """        

        # split csv string into list, map each list item to float,
        # then create list of floats
        int_list = list( map(int, csv.split(",")) )   
        return int_list     
