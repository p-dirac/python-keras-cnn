import pytest
from pytestqt.qt_compat import qt_api
from pytestqt.qtbot import QtBot
import numpy as np
import numpy.testing as npt
import array
#
from cnnpy.thenet.nettrain import NetTraining

class TestNetTraining():
    # no init() for Test class
    # but still need 'self' as first arg in each method

    @pytest.fixture
    def net(self):
        print("mat")
        nt = NetTraining(model=None, x_train=None, y_train=None, epochs=None, batch_size=None)
        return nt

    def testSizing(self, qtbot, qapp, net):
        # fixture 'net()' is not called as a method
        # the net arg is actually the return value of net()
        print("testSizing")
        numy = 50000
        epochs = 10
        batch_size = 20
        batches_per_epoch = numy/batch_size
        total= batches_per_epoch*epochs
        #
        net.sizing(numy, epochs, batch_size)
        assert net.total_batches == total

