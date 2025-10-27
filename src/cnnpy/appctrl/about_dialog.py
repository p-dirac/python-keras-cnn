#
import sys
import keras
import torch
import PySide6
from PySide6.QtWidgets import (
    QDialog, 
    QFormLayout, 
    QWidget,
    QLineEdit,
    QDialogButtonBox, 
    QVBoxLayout
)
from PySide6.QtGui import QFont
#from app_util import Util
from appctrl.app_util import Util

#
class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About CNN App")

        self.font = QFont("Arial", 12, QFont.Weight.Normal)
        # Create the form layout
        layout = self.prepAbout()
        widget = QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(700)

        # Create a standard button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Create the main vertical layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(widget)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)

    def prepAbout(self):
        layout = QFormLayout()
        # Add widgets to the layout
        #
        self.prep = QLineEdit()
        self.prep.setFont(self.font)
        app_version = Util.appVersion()
        self.prep.setText(app_version)
        layout.addRow("Application version", self.prep)
        #
        self.prep0 = QLineEdit()
        self.prep0.setFont(self.font)
        version_tuple = sys.version_info
        # version tuple has extra info, so don't use join(tuple)
        python_version = str(version_tuple.major) + "." + str(version_tuple.minor) + "." + str(version_tuple.micro)
        self.prep0.setText(python_version)
        layout.addRow("Python version", self.prep0)
        #
        self.prep1 = QLineEdit()
        self.prep1.setFont(self.font)
        self.prep1.setText(PySide6.__version__)
        layout.addRow("PySide6 version", self.prep1)
        #
        self.prep4 = QLineEdit()
        self.prep4.setFont(self.font)
        # Get the name of the current Keras backend
        backend_name = keras.backend.backend()
        self.prep4.setText(backend_name)
        layout.addRow("Keras backend name:", self.prep4)
        #
        self.prep2 = QLineEdit()
        self.prep2.setFont(self.font)
        device = torch.get_default_device()
        #print("torch device: ", device)
        if(str(device).startswith('cuda')):
            dev_name = "GPU: " + torch.cuda.get_device_name(0)
        else:
            dev_name = str(device)
        #
        #print("torch device name: ", dev_name)
        self.prep2.setText(str(dev_name))
        layout.addRow("Torch device name:", self.prep2)
        #
        return layout
    
