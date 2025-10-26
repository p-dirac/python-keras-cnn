import pytest
from pytestqt.qt_compat import qt_api
from pytestqt.qtbot import QtBot
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)
#
from cnnpy.app import AppWin

class TestAppWin():
    # no init() for Test class
    # but still need 'self' as first arg in each method

    @pytest.fixture
    def win(self, qtbot):
        print("widget qtbot")
        mw = AppWin()
        qtbot.addWidget(mw)
        return mw

    def testLoadParamsAct(self, qtbot, qapp, win):
        # fixture 'win()' is not called as a method
        # the win arg is actually the return value of win()
        print()
        print("testLoadParamsAct start")
        print("testLoadParamsAct menu bar", win.menuBar())
        # Find the 'loadParamsAct' action and trigger it
        loadParamsAct = None
        print("testLoadParamsAct actions: ", len(win.menuBar().actions()))
        for action in win.menuBar().actions():
            print("testLoadParamsAct, text: ", action.text())
            if action.text() == "App":
                for sub_action in action.menu().actions():
                    if sub_action.text() == "Load Data Params":
                        loadParamsAct = sub_action
                        break
            if loadParamsAct:
                break

        assert loadParamsAct is not None, "loadParamsAct not found"
        #loadParamsAct.trigger()

        # Assert that the status bar message changed
        #assert win.statusBar().currentMessage() == "Data Params loaded"

    def testLoadHyperAct(self, qtbot, qapp, win):
        # fixture 'win()' is not called as a method
        # the win arg is actually the return value of win()
        print()
        print("testLoadHyperAct start")
        # Find the 'loadParamsAct' action and trigger it
        loadHyperAct = None
        for action in win.menuBar().actions():
            print("testLoadHyperAct, action: ", action.text())
            if action.text() == "App":
                for sub_action in action.menu().actions():
                    print("testLoadHyperAct, sub_action: ", sub_action.text())
                    if sub_action.text() == "Load Hyper Params":
                        loadHyperAct = sub_action
                        break
            if loadHyperAct:
                break

        assert loadHyperAct is not None, "loadHyperAct not found"
        #loadHyperAct.trigger()

        # Assert that the status bar message changed
        #assert win.statusBar().currentMessage() == "Hyper Params loaded"
