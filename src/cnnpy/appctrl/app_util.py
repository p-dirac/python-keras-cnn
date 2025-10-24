#
import os
from pathlib import Path
import logging
logger = logging.getLogger()
#
from PySide6.QtWidgets import (
    QMessageBox,
    QFileDialog,
    QLayout,
    QWidget
)

class Util():
    """
    Utility of static methods.

    """
    def __init__(self):
        pass

    def __repr__(self):
        return "Util for AppWin"

    @staticmethod
    def alert(msg:str):
        btn = QMessageBox.critical(
        None,
        "Alert!",
        msg,
        buttons = QMessageBox.Ok,
        defaultButton=QMessageBox.Ok,)

    @staticmethod
    def openFileDialog(title: str, filters: list, dir: str):
        """Open dialog to read existing file

        Raises:
            Exception: if error occurs

        Returns:
            path: file path
        """        
        try:
            #print("openFileDialog, filters[0]: ", filters[0])
            fileName, open_filter = QFileDialog.getOpenFileName(None, title, str(dir), filter=filters[0])

            #print("openFileDialog, open_filter: ", open_filter)
            if fileName:    
                path = Path(fileName)
            else:
                path = None    

        except Exception:
            logger.error("Error in openFileDialog")
            Util.alert(f"Error in openFileDialog, for path:{path}")
            raise Exception("Error in openFileDialog") 
        else:    
            return path

    @staticmethod
    def saveFileDialog(title: str, filters: list, dir: str):
        """Opens dialog to save a file

        If file already exists, a confirm dialog will open

        Args:
            title (str): dialog title
            filters (list): type of file eg. *.json, *.h5

        Raises:
            Exception: if error occurs

        Returns:
            path: file path
        """        
        try:
            print("saveFileDialog, filters[0]: ", filters[0])
        
            fileName, save_filter = QFileDialog.getSaveFileName(None, title, str(dir), filter=filters[0])

            print("readParams, save_filter: ", save_filter)
            if fileName:    
                full = Path(fileName)
            else:
                full = None    

        except Exception:
            logger.error("Error in saveFileDialog")
            Util.alert(f"Error in saveFileDialog, for path:{full}")
            raise Exception("Error in saveFileDialog") 
        else:    
            return full

    @staticmethod
    def confirm(title: str, msg: str):
        # Show a confirmation message box
        confirmation = QMessageBox()
        confirmation.setStyleSheet("QLabel{min-width:500 px; font-size: 24px;} QPushButton{ width:150px; font-size: 18px; }")
        confirmation.setWindowTitle(title)
        confirmation.setText(msg)
        confirmation.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirmation.exec()
        if confirmation == QMessageBox.StandardButton.No:
            # User clicked No, so cancel the operation
            return -1
        else:
            # User clicked yes
            return 0

    @staticmethod
    def clearChildren(widget: QWidget):
        """
        Clears all child widgets from a given QWidget.

        Args:
            widget (QWidget): widget to be cleared

        """
        # layout(): returns layout manager (QLayout) on this widget
        if widget.layout():
            # count(): return number of items in the layout
            while widget.layout().count():
                # takeAt(0): remove the QLayoutItem at index 0 from the layout, # and return the item
                # other items will be renumbered
                child = widget.layout().takeAt(0)
                # child is QLayoutItem (a QWidget or QLayout)
                if child.widget():
                    # get QWidget 
                    child.widget().deleteLater()
                elif child.layout():
                    # get QLayout
                    # Recursively clear children of nested layouts
                    Util.clearLayout(child.layout())
        else:
            # no layout, iterate over children
            for child_widget in widget.findChildren(QWidget):
                child_widget.deleteLater()

    @staticmethod
    def clearLayout(layout: QLayout):
        """
        Clears all widgets within a given QLayout.

        Args:
            layout (QLayout): layout to be cleared

        """
        # count(): return number of items in the QLayout
        while layout.count():
            # takeAt(0): remove the QLayoutItem at index 0 from the layout, 
            # and return the item
            # other items will be renumbered
            # child is QLayoutItem (a QWidget or QLayout)
            child = layout.takeAt(0)
            if child.widget():
                # get QWidget
                child.widget().deleteLater()
            elif child.layout():
                # get QLayout
                # recursive call
                clearLayout(child.layout())

    


