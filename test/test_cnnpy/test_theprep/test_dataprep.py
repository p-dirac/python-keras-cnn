import pytest
from pytestqt.qt_compat import qt_api
from pytestqt.qtbot import QtBot
import numpy as np
import numpy.testing as npt
import array
#
from cnnpy.theprep.dataprep import DataPrep

class TestDataPrep():
    # no init() for Test class
    # but still need 'self' as first arg in each method

    @pytest.fixture
    def prep(self):
        print("mat")
        dp = DataPrep()
        return dp

    def testShuffleArrays(self, qtbot, qapp):
        # fixture 'prep()' is not called as a method
        # the prep arg is actually the return value of prep()
        print("testShuffleArrays")
        seed = 42
        a = np.array([1,2,3,4,5,6,7,8])
        # aa copy of a
        aa = array.array('i', a)
        b = np.array([101,102,103,104,105,106,107,108])
        # bb copy of b
        bb = array.array('i', b)
        print("before shuffle, a: \n", a)
        print("before shuffle, b: \n", b)
        # shuffle a, b in-place
        DataPrep.shuffleArrays(a, b, seed)
        print("after shuffle, a: \n", a)
        print("after shuffle, b: \n", b)
        # each array has changed by shuffling
        # "not equal" produces list of true,false
        # np.any check if any element in list is True
        assert np.any(a != aa)
        assert np.any(b != bb)
        # subtract a from b => all elements = 100
        # if shuffled together
        diff = np.subtract(b, a)
        # note: set allows only unique elements
        assert len(set(diff)) == 1 
        # pairs are preserved, but order changed
        # recall: zip combines pairs of elements into tuples
        # set is unordered
        unshuffled = set(zip(aa, bb))
        shuffled = set(zip(a, b))
        # compare two unordered sets that have same elements
        assert shuffled == unshuffled

