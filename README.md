This application demonstrates the implementation of a convolutional
neural network (CNN) programmed with Python class structures. A menu
user interface is provided by PySide6, while the network model is
created by Keras.

The application includes training, testing, and tuning tasks for
fitting the Keras network model. Matplotlib is employed to make plots
of accuracy, loss, confusion matrix, and sample dataset images. 
Gridsearchcv allows optimizaton across combinations of hyper-parameters.
Long running tasks are delegated to threads, with signal feedback to
update GUI progess bars. If CUDA is available, a GPU will speed up model
fitting.

Read the project description in Python-Keras-CNN.pdf found in the
docs folder.

Follow the installation instructions in uv-instructions.txt contained
in the docs folder.

