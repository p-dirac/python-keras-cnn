import os
import sys
os.environ["QT_API"] = "PySide6"
#
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as font_manager
#
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
#
import logging
logger = logging.getLogger()

#
class Plotter(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)

    def plotSamples(self, y_train: np.ndarray, x_train: np.ndarray):
        """Show samples of a few training labels and features

        Args:
            y_train (ndarray): training values of labels
            x_train (ndarray): training values of images

        Raises:
            Exception: if error occurs

        Returns:
            str: if no error 
        """
        try:
            #print("matplotlib backend: ", matplotlib.get_backend())
            self.fig.clear()
            # figsize: Width, height in inches.
            self.fig.set_size_inches(5, 4)
            self.fig.set_dpi(100)
            # display a few sample training images
            num_samples = 5  # per row
            # y_train should be ndarray
            # x_train type should be ndarray
            print("x_train shape:", x_train.shape)
            #
            # print a few cells
            #cells = x_train[1, 14, 1:, 0]
            # verify normalized image cells
            #print("plotSamples, x_train cells:", cells)
            #
            base = 0
            # for two rows, select only 2*num_samples images from x_train
            sample_images = x_train[base:base+2*num_samples]
            # check shape of single image
            print("sample_images[0] shape:", sample_images[0].shape)
            #
            # we want only the labels as an array
            actual_labels = y_train[base:base+2*num_samples]
            print("actual_labels: ", actual_labels)
            # GridSpec nrows, ncols
            gs = GridSpec(2, 5, figure=self.fig)
            #
            # fig size based on number of samples
            # two rows of samples
            for k in range(2):
                # number of cols = num_samples = 5
                for i in range(num_samples):
                    index = k*num_samples + i
                    #print(f"k: {k}, i: {i}, index:{index} ")
                    #
                    # subplot at row k, col i (gs is zero based)
                    ax = self.fig.add_subplot(gs[k, i])
                    #  
                    # show title
                    ax.set_title(f"Label: {actual_labels[index]}", fontfamily='Arial', fontsize=14, fontweight='bold')
                    #
                    # need to reshape image to proper format for plotting
                    # remove first dimension, to leave shape 
                    # of (w, h, c) where c = 1 or 3
                    image = sample_images[index].copy()
                    #
                    # Use cmap='gray' for grayscale images
                    # Note: shape of image must be (w, h, c) where c = 1 or 3
                    # imshow uses a normalized image
                    ax.imshow(image, cmap='gray') 
                    #
                    #Remove axis ticks and labels
                    ax.set_axis_off()
                    #
            # Draw the canvas 
            # Note: use self to call draw, not ax
            #print("plotSamples, call draw")
            self.fig.canvas.draw()
            #print("plotSamples, end samples plot")
        except Exception as ex:
            print("Error in plotSamples, ex: ", ex)
            logger.error("Error in plotSamples, ex: ", ex)
            raise Exception("Error in plotSamples") 
        else:    
            return "Completed plotSamples"

    def plotConfusion(self, cm: confusion_matrix, labels: list):    
        """Plot confusion matrix based on train or test data 

        Args:
            cm (ndarray) : confusion matrix, shape (num_classes, num_classes)
            labels (list) : label indexes 0 to num_classes-1, list length = num_classes

        Raises:
            Exception: if error occurs

        Returns:
            str: if no error 
        """

        try:
            #
            self.fig.clear()
            w = 6
            h = 6
            self.fig.set_size_inches(w, h)
            self.fig.set_dpi(90)
            #
            ax = self.fig.add_subplot(1, 1, 1)
            title_font = {'family':'Arial', 'size':'21', 'weight': 'bold'} 
            label_font = {'family':'Arial', 'size': '14', 'weight': 'bold'}
            ax.set_title('Confusion Matrix',fontdict=title_font)  
            #print("plotConfusion, begin plot")
            #
            # set two colors: first is for wrong prediction,
            # second color is for correct prediction
            offcmap = ListedColormap(['aliceblue'])
            diagcmap = ListedColormap(['white'])
            #
            # 2. Create masks for diagonal and off-diagonal elements
            # np.eye returns array with ones(true) on the diagonal and zeros(false) elsewhere.
            # asterisk * is the tuple unpacking operator. It unpacks the elements of the cm.shape tuple and passes them as separate positional arguments to the np.eye() function.
            # example: cm.shape is (4, 4), the call becomes np.eye(4, 4, dtype=bool)
            # Mask for off-diagonal elements (True for diagonal, False for off-diagonal)
            off_diag_mask = np.eye(*cm.shape, dtype=bool)

            #print("off_diag_mask: ", off_diag_mask)
            # Mask for diagonal elements (False for diagonal, True for off-diagonal)
            diag_mask = ~off_diag_mask

            # Plot off-diagonal elements with a gradient colormap (e.g., 'Reds')
            # mask: data will not be shown in cells where mask is True(on diag).
            h1 = sns.heatmap(cm, ax=ax, annot=True, cmap=offcmap, mask=off_diag_mask,
            fmt='d', cbar=False,  # Disable colorbar for off-diagonal
            linewidths=0.5, linecolor='k',
            xticklabels=labels, yticklabels=labels)

            # Plot diagonal elements with a solid color (e.g., 'Greens')
            # cm[off_diag_mask]: apply mask to cm to retrieve new array that only contains masked values.
            # mask: data will not be shown in cells where mask is True(off diag).
            h2 = sns.heatmap(cm, ax=ax, annot=True, cmap=diagcmap, mask=diag_mask,
            fmt='d', cbar=False,  # Disable colorbar for diagonal
            linewidths=0.5, linecolor='k',
            xticklabels=labels, yticklabels=labels)
            #
            # Issue with heatmap: right and bottom border not shown
            # Manually add a border around the entire heatmap
            ax.add_patch(Rectangle(
                (0, 0),  # (x, y) starting point
                cm.shape[1],  # width
                cm.shape[0],  # height
                fill=False,
                edgecolor='black',
                lw=0.5, # line width
                clip_on=False
            ))
            ax.set_xlabel('Predicted index', fontdict=label_font)
            ax.set_ylabel('True index', fontdict=label_font)
            ax.tick_params(axis='y', rotation=0)
            #
            # Adjust tick label font size
            ax.tick_params(axis='both', which='major', labelsize=12)
            # if grid lines on ticks, they go through the numbers being displayed; so don't use grid()
            #ax.grid(True)
            # Draw the canvas; 
            # recall: self is subclass of FigureCanvas
            self.fig.canvas.draw()
        except Exception as e:
            logger.error(f"Error in plotConfusion: {e}")
            print(f"Error in plotConfusion: {e}")
            raise Exception("Error in plotConfusion") 

        else:    
            return "Completed plotConfusion"



    def plotHistory(self, history):
        """Plot model accuracy vs epoch based on training data 

        Epoch list derived from length of accuracy list    
        
        Args:
            history (dict) : history contains two lists: accuracy, loss                  

        Raises:
            Exception: if error occurs

        Returns:
            str: if no error 
        """

        try:
            #
            self.fig.clear()
            self.fig.set_size_inches(6, 4)
            self.fig.set_dpi(90)
            # GridSpec nrows, ncols
            gs = GridSpec(1, 2, figure=self.fig)
            self.fig.subplots_adjust(left=0.10, right=0.90, bottom=0.2, top=.80, wspace=0.30, hspace=0.05)
            #
            #ax = self.fig.add_subplot(1, 1, 1)
            # subplot at row k, col i (gs is zero based)
            ax = self.fig.add_subplot(gs[0, 0])
            #
            #print("plotHistory, begin accuracy plot")
            #
            # 1st plot: accuracy
            ax.set_title('Training Accuracy', fontfamily='Arial', fontsize=16, fontweight='bold')
            # note history is a dict, plot x,y
            # where xe = epochs, yh = accuracy, label for legend
            # history dictionary contains two lists: accuracy, loss             
            yh = history['accuracy']
            # convert to percent
            yhp = [round(y*100,1) for y in yh]
            # number of epochs is length of accuracy list
            s = len(yh)
            # epochs begin at 1
            xe = list(range(1,s+1))
            ax.plot(xe, yhp, linestyle='solid', label='train', linewidth=3)
            ax.set_xlabel("Epoch", fontfamily='Arial', fontsize=14, fontweight='bold')
            ax.set_ylabel("Accuracy %", fontfamily='Arial', fontsize=14, fontweight='bold')
            font_properties = font_manager.FontProperties(family='Arial', size=12, weight='bold')
            ax.legend(prop=font_properties)
            #print("plotHistory, end accuracy plot")
            #
            #print("plotHistory, begin loss plot")
            #
            # subplot at row k, col i (gs is zero based)
            ax = self.fig.add_subplot(gs[0, 1])
            #
            # 2nd plot: loss
            ax.set_title('Model loss', fontfamily='Arial', fontsize=16, fontweight='bold')
            # note history is a dict, plot x,y
            # where x = epochs, y = loss
            yh = history['loss']
            s = len(yh)
            # number of epochs is length of loss list
            s = len(yh)
            # epochs begin at 1
            xe = list(range(1,s+1))
            ax.plot(xe, yh, linestyle='solid', label='train', linewidth=3)
            ax.set_xlabel('Epoch', fontfamily='Arial', fontsize=14, fontweight='bold')
            ax.set_ylabel('Loss', fontfamily='Arial', fontsize=14, fontweight='bold')
            ax.legend(prop=font_properties)
            # Draw the canvas; 
            # recall: self is subclass of FigureCanvas
            # Note: use self to call draw, not ax
            #print("plotHistory, end loss plot")
            self.fig.canvas.draw()

        except Exception:
            logger.error("Error in plotHistory")
            raise Exception("Error in plotHistory") 
        else:    
            return self.fig, ax

    def plotHistoryVal(self, history):
        """Plot model accuracy vs epoch based on training data 

        Epoch list derived from length of accuracy list

        Args:
            history (dict) : history contains four lists: accuracy, val_accuracy, loss, val_loss                  

        Raises:
            Exception: if error occurs

        Returns:
            FigureCanvas, Axes : the figure and its axes
        """

        try:
            # recall self.fig = Figure()
            self.fig.clear()
            self.fig.set_size_inches(6, 4)
            self.fig.set_dpi(90)
            # GridSpec nrows, ncols
            gs = GridSpec(1, 2, figure=self.fig)
            self.fig.subplots_adjust(left=0.10, right=0.90, bottom=0.2, top=.80, wspace=0.30, hspace=0.05)
            #
            #ax = self.fig.add_subplot(1, 1, 1)
            # subplot at row k, col i (gs is zero based)
            ax = self.fig.add_subplot(gs[0, 0])
            #
            #print("plotHistory, begin accuracy plot")
            #
            # 1st plot: accuracy
            ax.set_title("Training/Validation Accuracy", fontfamily='Arial', fontsize=16, fontweight='bold')
            # where x = epochs, y = accuracy
            # history dictionary contains four lists: accuracy, val_accuracy, loss, val_loss 
            yh = history['accuracy']
            yv = history['val_accuracy']
            # convert to percent
            yhp = [round(y*100,1) for y in yh]
            yvp = [round(y*100,1) for y in yv]
            # number of epochs is length of accuracy list
            s = len(yh)
            # epochs begin at 1
            xe = list(range(1,s+1))
            #print("plotHistory, acc type:", type(yh))
            #print("plotHistory, acc xe:", xe)
            #print("plotHistory, acc:", yh)
            #print("plotHistory, val acc:", yv)
            ax.plot(xe, yhp, linestyle='solid', label='Training Accuracy', linewidth=3)
            ax.plot(xe, yvp, linestyle='solid', label='Validation Accuracy', linewidth=3)
            ax.set_xlabel("Epoch", fontfamily='Arial', fontsize=14, fontweight='bold')
            ax.set_ylabel("Accuracy %", fontfamily='Arial', fontsize=14, fontweight='bold')
            font_properties = font_manager.FontProperties(family='Arial', size=12, weight='bold')
            ax.legend(prop=font_properties)
            #print("plotHistory, end accuracy plot")
            #
            #print("plotHistory, begin loss plot")
            #
            # subplot at row k, col i (gs is zero based)
            ax = self.fig.add_subplot(gs[0, 1])
            #
            # 2nd plot: loss
            ax.set_title('Model loss', fontfamily='Arial', fontsize=16, fontweight='bold')
            # where x = epochs, y = loss
            # history contains four lists: accuracy, val_accuracy, loss, val_loss 
            yh = history['loss']
            yv = history['val_loss']
            s = len(yh)
            # number of epochs is length of loss list
            s = len(yh)
            # epochs begin at 1
            xe = list(range(1,s+1))
            #print("plotHistory, loss xe:", xe)
            #print("plotHistory, loss:", yh)
            #print("plotHistory, val loss:", yv)
            ax.plot(xe, yh, linestyle='solid', label='Training Loss', linewidth=3)
            ax.plot(xe, yv, linestyle='solid', label='Validation Loss', linewidth=3)
            ax.set_xlabel('Epoch', fontfamily='Arial', fontsize=14, fontweight='bold')
            ax.set_ylabel('Loss', fontfamily='Arial', fontsize=14, fontweight='bold')
            ax.legend(prop=font_properties)
            # Draw the canvas; 
            # recall: self is subclass of FigureCanvas
            # Note: use self to call draw, not ax
            self.fig.canvas.draw()
            #self.draw()
            #print("plotHistory, end loss plot")

        except Exception as ex:
            print("Error in plotHistory, ex:", ex)
            logger.error("Error in plotHistory")
            raise Exception("Error in plotHistory") 
        else: 
            # for testing only    
            return self.fig, ax
