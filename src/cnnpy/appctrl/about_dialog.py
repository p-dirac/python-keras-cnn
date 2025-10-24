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
#
class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About CNN App")

        self.font = QFont("Arial", 14, QFont.Weight.Bold)
        # Create the form layout
        layout = self.prepAbout()
        widget = QWidget()
        widget.setLayout(layout)
        widget.setFixedWidth(600)

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
        self.prep2 = QLineEdit()
        self.prep2.setFont(self.font)
        gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu)
        self.prep2.setText(gpu_name)
        layout.addRow("GPU name:", self.prep2)
        #
        self.prep3 = QLineEdit()
        self.prep3.setFont(self.font)
        num_gpus = torch.cuda.device_count()
        self.prep3.setText(str(num_gpus))
        layout.addRow("Number of GPUs:", self.prep3)
        #
        self.prep4 = QLineEdit()
        self.prep4.setFont(self.font)
        # Get the name of the current Keras backend
        backend_name = keras.backend.backend()
        self.prep4.setText(backend_name)
        layout.addRow("Backend name:", self.prep4)
        #
        return layout
    
