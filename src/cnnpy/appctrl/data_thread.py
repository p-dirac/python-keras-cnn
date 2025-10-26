#
from PySide6.QtCore import QThread, Signal
from theprep.dataprep import DataPrep
#
class DataTask(QThread):
    #
    # Note: only code inside run() will execute in a separate thread
    # That thread will exit after the run() function has returned
    #
    # Signal for the window to establish the maximum value
    # of the progress bar.
    setMaxProgress = Signal(float)
    # Signal to increase the progress.
    setUpdateProgress = Signal(float)
    #
    labels = None
    features = None
    subtotal = 0

    def __init__(self, dataPrep: DataPrep, data_dir: str):
        # must include super call for QObject
        super().__init__()
        """
        Args:
            dataPrep (DataPrep): class to read, shuffle, and normalize neural network data
            data_dir (str): data directory parent
        """
        self.dataPrep = dataPrep
        self.num_classes = self.dataPrep.num_classes
        self.image_size = self.dataPrep.image_size
        self.data_dir = data_dir

    def updateProgress(self, k):
        # ignore k, just need to update subtotal on each call
        self.subtotal += 1 
        r = self.subtotal/(self.num_classes)
        update = r*100
        #print("progress update:", update) 
        # sends signal to the GUI
        self.setUpdateProgress.emit(update)


    def run(self):
        # Note: only code inside run() will execute in a separate thread
        # load data, shuffle, and normalize
        # the callback, updateProgress, sends signal to the GUI
        self.labels, self.features = self.dataPrep.runTask(self.data_dir, self.updateProgress)


