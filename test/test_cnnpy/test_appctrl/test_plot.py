import pytest
from pytestqt.qt_compat import qt_api
from pytestqt.qtbot import QtBot
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)
import numpy as np
#
from cnnpy.app import AppWin
from cnnpy.appctrl.plotter import Plotter

class TestAppWin():
    # no init() for Test class
    # but still need 'self' as first arg in each method

    @pytest.fixture
    def win(self, qtbot):
        print("widget qtbot")
        mw = AppWin()
        qtbot.addWidget(mw)
        return mw
    
    def testPlotHistoryVal(self):
        mp = Plotter()
        # x = epochs (1 to 8), d1 = accuracy, d2 = val_accuracy
        # history accuracy is a decimal fraction
        y1 = [50, 55, 58, 60, 70, 75, 80, 95]
        y1f = [y / 100 for y in y1]
        y2 = [60, 65, 68, 70, 80, 85, 90, 98]
        y2f = [y / 100 for y in y2]
        # losses
        y3 = [2, .8, .5, .3, .2, .1, .05, .04]
        y4 = [3, 2, .7, .5, .2, .05, .02, .01]
        # history contains four lists: accuracy, val_accuracy, loss, val_loss
        # history accuracy is a decimal fraction
        history = dict(accuracy=y1f, val_accuracy=y2f, loss=y3, val_loss=y4)
        print("begin testPlot")
        print("testPlot, y1: \n", y1)
        print("testPlot, y1f: \n", y1f)
        print("testPlot, y2: \n", y2)
        print("testPlot, y2f: \n", y2f)
        print("testPlot, history: \n", history)
        fig, ax = mp.plotHistoryVal(history)
        # Assert the data in the plot's lines
        # plot accuracy is percent
        assert np.array_equal(fig.axes[0].lines[0].get_ydata(), y1)
        assert np.array_equal(fig.axes[0].lines[1].get_ydata(), y2)
        # losses
        #assert np.array_equal(fig.axes[0].lines[2].get_ydata(), y3)
        #assert np.array_equal(fig.axes[0].lines[3].get_ydata(), y4)
        #mp.close(fig) # Close the figure to prevent resource leaks
