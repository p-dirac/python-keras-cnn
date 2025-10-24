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
    QPlainTextEdit,
    QTabWidget,
    QDialogButtonBox, 
    QVBoxLayout
)
from PySide6.QtGui import QFont
import textwrap
from textwrap import dedent
#
class TutorDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("CNN App Tutorial")

        self.font = QFont("Arial", 14, QFont.Weight.Bold)
        # Create the layout
        layout = self.prepTutor()
        widget = QWidget()
        widget.setLayout(layout)
        widget.setFixedSize(900, 700)
        #widget.setFixedWidth(800)

        # Create a standard button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Create the main vertical layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(widget)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)

    def prepTutor(self):
        #
        layout = self.createTabs()
        #
        self.addText1()
        self.addText2()
        self.addText3()
        self.addText4()
        #
        return layout

    def createTabs(self):
        vlayout = QVBoxLayout()
        #
        # Create tab widget with tabs
        self.tabs = QTabWidget()
        #self.tabs.currentChanged.connect(self.tabChanged)
        self.tabs.setFont(self.font)
        vlayout.addWidget(self.tabs)
        return vlayout

    def addText1(self):
        # triple quoted string has hidden newlines
        tx = """
        Scenario 1: Training and Testing
        Load Data Params (select mnist-data-params.json)
        Load Hyper Params (select mnist-hyper-params-5.json)
        Load Training Dataset
        Initialize New Network Model
        Training with Validation (observe accuracy, loss plots)
        (Optional) Under help menu, Snap to record image of accuracy, loss plots
        ...
        Load Testing Dataset
        Evaluation (compare accuracy with training plot)
        Prediction (inspect off diagonal incorrect values)
        """
        vlayout = self.prepText(tx)
        self.addTab("Scenario 1", vlayout)

    def addText2(self):
        tx = """
        Scenario 2: Testing from Model
        Load Data Params (select mnist-data-params.json)
        Load Network Model (select mnist-model.keras)
        Load Testing Dataset
        Evaluation
        Prediction (inspect off diagonal incorrect values)
        """
        vlayout = self.prepText(tx)
        self.addTab("Scenario 2", vlayout)

    def addText3(self):
        tx = """
        Scenario 3: Tuning
        Load Data Params (select mnist-data-params.json)
        Load Hyper Params (select mnist-hyper-params-5.json)
        Load Training Dataset
        Edit Hyper Params (optional)
        Tuning (scroll through grid search results, and note best set of hyper parameters)
        (Optional) Under help menu, Snap to record image of grid search results
        Stop Tuning Task (optional)
        """
        vlayout = self.prepText(tx)
        self.addTab("Scenario 3", vlayout)

    def addText4(self):
        # triple quoted string has hidden newlines
        tx = """
        Scenario 4: Training and Manual Tuning
        Load Data Params (select mnist-data-params.json)
        Load Hyper Params (select mnist-hyper-params-5.json)
        Load Training Dataset
        Load Testing Dataset
        ...
        Edit Hyper Params
        Initialize New Network Model (must do this to use edited hyper params)
        Training with Validation (observe accuracy plots)
        Evaluation (note accuracy percentage e.g. 99.05)
        Save Hyper Params (select a file, e.g. mnist-hyper-acc-99-05, to associate testing accuracy with current hyper params set)
        Repeat these steps by editing the hyper params
        ...
        For final trained model:
        Save Network Model (may be used in Scenario 2)
        """
        vlayout = self.prepText(tx)
        self.addTab("Scenario 4", vlayout)

    def prepText(self, tx):
        #
        prep = QPlainTextEdit()
        prep.setReadOnly(True)
        # remove leading whitespace on each line of tx
        lines = textwrap.dedent(tx)
        prep.setPlainText(lines)
        prep.setFont(self.font)
        #
        vlayout = QVBoxLayout()
        vlayout.addWidget(prep)
        #
        return vlayout

    def addTab(self, name: str, prepLayout: QVBoxLayout):
        prepTab = QWidget()
        prepTab.setFont(self.font)
        prepTab.setLayout(prepLayout)
        self.tabs.addTab(prepTab, name)


