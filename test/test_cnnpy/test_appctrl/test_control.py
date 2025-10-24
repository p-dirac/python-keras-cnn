import pytest
from pytestqt.qt_compat import qt_api
from pytestqt.qtbot import QtBot
import numpy as np
import numpy.testing as npt

#
from cnnpy.app import AppWin
from cnnpy.appctrl.controller import Controller

class TestController():
    # no init() for Test class
    # but still need 'self' as first arg in each method
    

    @pytest.fixture
    def app(self):
        win = AppWin()
        return win


    def testUpdateStatus(self, qtbot, qapp, app):
        # fixture 'ctrl()' is not called as a method
        # the ctrl arg is actually the return value of ctrl()
        msg = "Training underway"
        ctrl = app.getController()
        # Create a SignalBlocker to wait (in msec) for the signal
        # Wait for the ctrl.updateStatus signal
        with qtbot.waitSignal(ctrl.updateStatus, timeout=1000) as blocker:
            # trigger a signal that the test is waiting for
            ctrl.updateStatus.emit(msg)

        # Assert that the signal was triggered
        assert blocker.signal_triggered  
        # Assert the arguments received with the signal
        assert blocker.args == [msg] 
        # check if signal received in AppWin
        m = app.statusBar().currentMessage()
        assert m == msg

