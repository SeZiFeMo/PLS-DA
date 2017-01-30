#!/usr/bin/env python
# coding: utf-8

import IO
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import sys


def new_qt(widget, name, parent=None):
    """Return a widget of type specified with 1st argument (a string).

       Also set the object name and optionally the parent widjet.
    """
    ret = getattr(QtWidgets, widget)(parent)
    ret.setObjectName(name)
    return ret


def setSize(widget, minimum=None, maximum=None, base=None):
    """Set the minimum, maximum and base sizes.

       Each one can be a list or tuple which must have two integer values:
         horizontal and vertical between 0 and the UHDTV size.
    """
    for sizes, attr in ((minimum, 'setMinimumSize'),
                        (maximum, 'setMaximumSize'),
                        (base, 'setBaseSize')):
        if sizes is not None and \
           (isinstance(sizes, tuple) or isinstance(sizes, list)) and \
           len(sizes) == 2:
            sizes = tuple(map(int, sizes))
            if sizes[0] in range(0, 7681) and sizes[1] in range(0, 4321):
                getattr(widget, attr)(QtCore.QSize(*sizes))
            else:
                func = '.'.join(widget.objectName(), attr).lstrip('.')
                IO.Log.warning('{}({},{}) failed! Width not in [0; 7680] or '
                               'height not in [0; 4320]'.format(func, *sizes))


def setPolicy(widget, h_policy='Preferred', v_policy='Preferred',
              h_stretch_factor=0, v_stretch_factor=0):
    """Set the new size policy of widget.

       widget is used to keep the previous hasHeightForWidth value and
         to set on it the new size policy.
       set [h|v]_stretch_factor to None to avoid setting it.
    """
    addmitted_size_policies = ('Fixed', 'Minimum', 'Maximum', 'Preferred',
                               'Expanding', 'MinimumExpanding', 'Ignored')
    if h_policy not in addmitted_size_policies or \
       v_policy not in addmitted_size_policies:
        IO.Log.error('Unknown size policy ({}, {})'.format(h_policy, v_policy))
        exit(1)

    size_policy = QtWidgets.QSizePolicy(getattr(QtWidgets.QSizePolicy,
                                                h_policy),
                                        getattr(QtWidgets.QSizePolicy,
                                                v_policy))
    if h_stretch_factor is not None:
        size_policy.setHorizontalStretch(h_stretch_factor)
    if v_stretch_factor is not None:
        size_policy.setVerticalStretch(v_stretch_factor)

    """Was the previous widget preferred height depending on its width?
       Lets keep the same!
    """
    size_policy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    widget.setSizePolicy(size_policy)


class UserInterface(object):

    drop_down_choices = ['Scree', 'LVs - Explained variance Y',
                         'Inner relationships', 'Biplot', 'Scores & Loadings',
                         'Scores', 'Loadings', 'Samples - Y calculated',
                         'Samples - Y predicted', 'T2 - Q',
                         'Residuals - Leverage', 'Regression coefficients']

    def __init__(self):
        self.MainWindow = new_qt('QMainWindow', 'MainWindow')
        self.MainWindow.setEnabled(True)
        self.MainWindow.resize(800, 600)
        setPolicy(self.MainWindow, 'Expanding', 'Expanding', 0, 0)
        setSize(self.MainWindow, minimum=(800, 600), maximum=(7680, 4320))
        self.MainWindow.setUnifiedTitleAndToolBarOnMac(True)

        # Previously in setupUi()
        self.MainWidget = new_qt('QWidget', 'MainWidget', parent=self.MainWindow)
        setPolicy(self.MainWidget, 'Preferred', 'Preferred', 0, 0)
        setSize(self.MainWidget, minimum=(800, 600), maximum=(7680, 4300))

        self.MainSplitter = new_qt('QSplitter', 'MainSplitter', parent=self.MainWidget)
        setPolicy(self.MainSplitter, 'Expanding', 'Expanding', 0, 0)
        setSize(self.MainSplitter, minimum=(800, 600), maximum=(7680, 4300))
        self.MainSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.MainSplitter.setHandleWidth(3)

        # Start creating widgets to put inside LeftWidget
        self.LeftScrollAreaWidgetContents = new_qt('QWidget',
                                                'LeftScrollAreaWidgetContents')
        self.LeftScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 291, 565))
        setPolicy(self.LeftScrollAreaWidgetContents, 'Expanding', 'Expanding', 0, 0)
        setSize(self.LeftScrollAreaWidgetContents, minimum=(174, 427), maximum=(3611, 4147))
        self.LeftScrollAreaWidgetContents.setLayoutDirection(QtCore.Qt.LeftToRight)

        self.PlotFormLayout = new_qt('QFormLayout', 'PlotFormLayout', parent=self.LeftScrollAreaWidgetContents)
        self.PlotFormLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.PlotFormLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.PlotFormLayout.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.PlotFormLayout.setFormAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.PlotFormLayout.setContentsMargins(10, 10, 10, 10)
        self.PlotFormLayout.setSpacing(10)

        self.LeftLVsLabel = new_qt('QLabel', 'LeftLVsLabel', parent=self.LeftScrollAreaWidgetContents)
        setSize(self.LeftLVsLabel, minimum=(70, 22), maximum=(1310, 170))
        self.LeftLVsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.LeftLVsLabel.setWordWrap(True)
        self.PlotFormLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.LeftLVsLabel)

        self.LeftLVsSpinBox = new_qt('QSpinBox', 'LeftLVsSpinBox', parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftLVsSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftLVsSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftLVsSpinBox.setMinimum(1)
        self.PlotFormLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.LeftLVsSpinBox)

        self.LeftXRadioButton = new_qt('QRadioButton', 'LeftXRadioButton', parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftXRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftXRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.PlotFormLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.LeftXRadioButton)

        self.LeftYRadioButton = new_qt('QRadioButton', 'LeftYRadioButton', parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftYRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftYRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.PlotFormLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.LeftYRadioButton)

        self.LeftButtonGroup = new_qt('QButtonGroup', 'LeftButtonGroup', parent=self.MainWindow)
        self.LeftButtonGroup.addButton(self.LeftXRadioButton)
        self.LeftButtonGroup.addButton(self.LeftYRadioButton)

        self.LeftXSpinBox = new_qt('QSpinBox', 'LeftXSpinBox', parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftXSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftXSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftXSpinBox.setMinimum(1)
        self.PlotFormLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.LeftXSpinBox)

        self.LeftYSpinBox = new_qt('QSpinBox', 'LeftYSpinBox', parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftYSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftYSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftYSpinBox.setMinimum(1)
        self.PlotFormLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.LeftYSpinBox)

        self.LeftPlotPushButton = new_qt('QPushButton', 'LeftPlotPushButton', parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftPlotPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftPlotPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.PlotFormLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.LeftPlotPushButton)

        self.LeftBackPushButton = new_qt('QPushButton', 'LeftBackPushButton', parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftBackPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftBackPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.PlotFormLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.LeftBackPushButton)

        self.LeftWidget = new_qt('QWidget', 'LeftWidget', parent=self.MainSplitter)
        setSize(self.LeftWidget, minimum=(200, 580), maximum=(3637, 4300))

        self.LeftComboBox = new_qt('QComboBox', 'LeftComboBox', parent=self.LeftWidget)
        setSize(self.LeftComboBox, minimum=(194, 22), maximum=(3631, 22))
        for entry in self.drop_down_choices:
            self.LeftComboBox.addItem("")

        self.LeftScrollArea = new_qt('QScrollArea', 'LeftScrollArea', parent=self.LeftWidget)
        setSize(self.LeftScrollArea, minimum=(194, 547), maximum=(3631, 4267))
        self.LeftScrollArea.setWidgetResizable(True)
        self.LeftScrollArea.setWidget(self.LeftScrollAreaWidgetContents)

        self.LeftGridLayout = new_qt('QGridLayout', 'LeftGridLayout', parent=self.LeftWidget)
        self.LeftGridLayout.setContentsMargins(3, 3, 3, 3)
        self.LeftGridLayout.setSpacing(5)
        self.LeftGridLayout.addWidget(self.LeftComboBox, 0, 0, 1, 1)
        self.LeftGridLayout.addWidget(self.LeftScrollArea, 1, 0, 1, 1)

        # Start creating widgets to put inside CentralWidget
        self.CentralScrollAreaWidgetContents = new_qt('QWidget', 'CentralScrollAreaWidgetContents')
        self.CentralScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 290, 565))
        setPolicy(self.CentralScrollAreaWidgetContents, 'Expanding', 'Expanding', 0, 0)
        setSize(self.CentralScrollAreaWidgetContents, minimum=(174, 427), maximum=(3611, 4147))
        self.CentralScrollAreaWidgetContents.setLayoutDirection(QtCore.Qt.LeftToRight)

        self.PlotFormLayout1 = new_qt('QFormLayout', 'PlotFormLayout1', parent=self.CentralScrollAreaWidgetContents)
        self.PlotFormLayout1.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.PlotFormLayout1.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.PlotFormLayout1.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.PlotFormLayout1.setFormAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.PlotFormLayout1.setContentsMargins(10, 10, 10, 10)
        self.PlotFormLayout1.setSpacing(10)

        self.CentralLVsLabel = new_qt('QLabel', 'CentralLVsLabel', parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralLVsLabel, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralLVsLabel, minimum=(70, 22), maximum=(1310, 170))
        self.CentralLVsLabel.setTextFormat(QtCore.Qt.AutoText)
        self.CentralLVsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.CentralLVsLabel.setWordWrap(True)
        self.PlotFormLayout1.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.CentralLVsLabel)

        self.CentralLVsSpinBox = new_qt('QSpinBox', 'CentralLVsSpinBox', parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralLVsSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralLVsSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralLVsSpinBox.setMinimum(1)
        self.PlotFormLayout1.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.CentralLVsSpinBox)

        self.CentralXRadioButton = new_qt('QRadioButton', 'CentralXRadioButton', parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralXRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralXRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.PlotFormLayout1.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.CentralXRadioButton)

        self.CentralYRadioButton = new_qt('QRadioButton', 'CentralYRadioButton', parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralYRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralYRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.PlotFormLayout1.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.CentralYRadioButton)

        self.CentralButtonGroup = new_qt('QButtonGroup', 'CentralButtonGroup', parent=self.MainWindow)
        self.CentralButtonGroup.addButton(self.CentralXRadioButton)
        self.CentralButtonGroup.addButton(self.CentralYRadioButton)

        self.CentralXSpinBox = new_qt('QSpinBox', 'CentralXSpinBox', parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralXSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralXSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralXSpinBox.setMinimum(1)
        self.PlotFormLayout1.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.CentralXSpinBox)

        self.CentralYSpinBox = new_qt('QSpinBox', 'CentralYSpinBox', parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralYSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralYSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralYSpinBox.setMinimum(1)
        self.PlotFormLayout1.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.CentralYSpinBox)

        self.CentralBackPushButton = new_qt('QPushButton', 'CentralBackPushButton',
                parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralBackPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralBackPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.PlotFormLayout1.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.CentralBackPushButton)

        self.CentralPlotPushButton = new_qt('QPushButton', 'CentralPlotPushButton',
                parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralPlotPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralPlotPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.PlotFormLayout1.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.CentralPlotPushButton)

        self.CentralWidget = new_qt('QWidget', 'CentralWidget', parent=self.MainSplitter)
        setSize(self.CentralWidget, minimum=(200, 580), maximum=(3637, 4300))

        self.CentralComboBox = new_qt('QComboBox', 'CentralComboBox', parent=self.CentralWidget)
        setPolicy(self.CentralComboBox, 'Expanding', 'Expanding', 0, 0)
        setSize(self.CentralComboBox, minimum=(194, 22), maximum=(3631, 22))
        for entry in self.drop_down_choices:
            self.CentralComboBox.addItem("")

        self.CentralScrollArea = new_qt('QScrollArea', 'CentralScrollArea', parent=self.CentralWidget)
        setSize(self.CentralScrollArea, minimum=(194, 547), maximum=(3631, 4267))
        self.CentralScrollArea.setWidgetResizable(True)
        self.CentralScrollArea.setWidget(self.CentralScrollAreaWidgetContents)

        self.CentralGridLayout = new_qt('QGridLayout', 'CentralGridLayout', parent=self.CentralWidget)
        self.CentralGridLayout.setContentsMargins(3, 3, 3, 3)
        self.CentralGridLayout.setSpacing(5)
        self.CentralGridLayout.addWidget(self.CentralComboBox, 0, 0, 1, 1)
        self.CentralGridLayout.addWidget(self.CentralScrollArea, 1, 0, 1, 1)

        # Start creating widgets to put inside RightWidget
        self.RightScrollAreaWidgetContents = new_qt('QWidget', 'RightScrollAreaWidgetContents')
        self.RightScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 189, 565))
        setPolicy(self.RightScrollAreaWidgetContents, 'Expanding', 'Expanding', 0, 0)
        setSize(self.RightScrollAreaWidgetContents, minimum=(138, 534), maximum=(388, 4259))

        self.DetailsLabel = new_qt('QLabel', 'DetailsLabel', parent=self.RightScrollAreaWidgetContents)
        setPolicy(self.DetailsLabel, 'Expanding', 'Expanding', 0, 0)
        setSize(self.DetailsLabel, minimum=(138, 534), maximum=(388, 4259))
        self.DetailsLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.DetailsLabel.setWordWrap(True)
        self.DetailsLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)

        self.gridLayout = new_qt('QGridLayout', 'gridLayout', parent=self.RightScrollAreaWidgetContents)
        self.gridLayout.setContentsMargins(3, 3, 3, 3)
        self.gridLayout.setSpacing(5)
        self.gridLayout.addWidget(self.DetailsLabel, 0, 0, 1, 1)

        self.RightWidget = new_qt('QWidget', 'RightWidget', parent=self.MainSplitter)
        setPolicy(self.RightWidget, 'Expanding', 'Expanding', 0, 0)
        setSize(self.RightWidget, minimum=(150, 580), maximum=(400, 4300))

        self.RightScrollArea = new_qt('QScrollArea', 'RightScrollArea', parent=self.RightWidget)
        setSize(self.RightScrollArea, minimum=(144, 547), maximum=(394, 4272))
        self.RightScrollArea.setWidgetResizable(True)
        self.RightScrollArea.setWidget(self.RightScrollAreaWidgetContents)

        self.CurrentModeLabel = new_qt('QLabel', 'CurrentModeLabel', parent=self.RightWidget)
        setPolicy(self.CurrentModeLabel, 'Expanding', 'Expanding', 0, 0)
        setSize(self.CurrentModeLabel, minimum=(144, 22), maximum=(394, 22))
        self.CurrentModeLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.CurrentModeLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.CurrentModeLabel.setLineWidth(1)
        self.CurrentModeLabel.setTextFormat(QtCore.Qt.AutoText)
        self.CurrentModeLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.RightGridLayout = new_qt('QGridLayout', 'RightGridLayout', parent=self.RightWidget)
        self.RightGridLayout.setContentsMargins(3, 3, 3, 3)
        self.RightGridLayout.setSpacing(5)
        self.RightGridLayout.addWidget(self.RightScrollArea, 1, 0, 1, 1)
        self.RightGridLayout.addWidget(self.CurrentModeLabel, 0, 0, 1, 1)

        self.MainGridLayout = new_qt('QGridLayout', 'MainGridLayout', parent=self.MainWidget)
        self.MainGridLayout.setContentsMargins(0, 0, 0, 0)
        self.MainGridLayout.setSpacing(0)
        self.MainGridLayout.addWidget(self.MainSplitter, 0, 0, 1, 1)
        self.MainWindow.setCentralWidget(self.MainWidget)

        self.TopMenuBar = new_qt('QMenuBar', 'TopMenuBar', parent=self.MainWindow)
        self.TopMenuBar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        setSize(self.TopMenuBar, minimum=(800, 20), maximum=(7680, 20))

        self.MenuOptions = new_qt('QMenu', 'MenuOptions', parent=self.TopMenuBar)
        setSize(self.MenuOptions, minimum=(100, 20), maximum=(960, 4300))

        self.MenuChangeMode = new_qt('QMenu', 'MenuChangeMode', parent=self.TopMenuBar)
        setSize(self.MenuChangeMode, minimum=(100, 20), maximum=(960, 4300))

        self.MenuAbout = new_qt('QMenu', 'MenuAbout', parent=self.TopMenuBar)
        setSize(self.MenuAbout, minimum=(100, 20), maximum=(960, 4300))

        self.MainWindow.setMenuBar(self.TopMenuBar)

        qa, mw = 'QAction', self.MainWindow
        self.ActionExport = new_qt(qa, 'ActionExport', parent=mw)
        self.ActionModel = new_qt(qa, 'ActionModel', parent=mw)
        self.ActionCV = new_qt(qa, 'ActionCV', parent=mw)
        self.ActionPrediction = new_qt(qa, 'ActionPrediction', parent=mw)
        self.ActionQuit = new_qt(qa, 'ActionQuit', parent=mw)
        self.ActionSaveModel = new_qt(qa, 'ActionSaveModel', parent=mw)
        self.ActionLoadModel = new_qt(qa, 'ActionLoadModel', parent=mw)
        self.ActionNewModel = new_qt(qa, 'ActionNewModel', parent=mw)
        self.ActionAboutThatProject = new_qt(qa, 'ActionAboutThatProject', parent=mw)

        self.MenuOptions.addAction(self.ActionNewModel)
        self.MenuOptions.addAction(self.ActionSaveModel)
        self.MenuOptions.addAction(self.ActionLoadModel)
        self.MenuOptions.addSeparator()
        self.MenuOptions.addAction(self.ActionExport)
        self.MenuOptions.addSeparator()
        self.MenuOptions.addAction(self.ActionQuit)

        self.MenuChangeMode.addAction(self.ActionModel)
        self.MenuChangeMode.addAction(self.ActionCV)
        self.MenuChangeMode.addAction(self.ActionPrediction)

        self.MenuAbout.addAction(self.ActionAboutThatProject)

        self.TopMenuBar.addAction(self.MenuOptions.menuAction())
        self.TopMenuBar.addAction(self.MenuChangeMode.menuAction())
        self.TopMenuBar.addAction(self.MenuAbout.menuAction())

        # Previously in retranslateUi()
        self.MainWindow.setWindowTitle("PLS-DA")

        for index, entry in enumerate(self.drop_down_choices):
            self.LeftComboBox.setItemText(index, entry)
            self.CentralComboBox.setItemText(index, entry)

        self.LeftLVsLabel.setText("Latent Variables")
        self.LeftXRadioButton.setText("X")
        self.LeftYRadioButton.setText("Y")
        self.LeftPlotPushButton.setText("Plot")
        self.LeftBackPushButton.setText("Back")

        self.CentralLVsLabel.setText("Latent Variables")
        self.CentralXRadioButton.setText("X")
        self.CentralYRadioButton.setText("Y")
        self.CentralBackPushButton.setText("Back")
        self.CentralPlotPushButton.setText("Plot")

        self.DetailsLabel.setText("Details")
        self.currentMode('start')

        self.ActionAboutThatProject.setText("A&bout this project")
        self.ActionCV.setText("Cross&Validation")
        self.ActionExport.setText("&Export matrices")
        self.ActionLoadModel.setText("&Load model")
        self.ActionModel.setText("&Model")
        self.ActionNewModel.setText("&New model")
        self.ActionPrediction.setText("&Prediction")
        self.ActionQuit.setText("&Quit")
        self.ActionSaveModel.setText("&Save model")

        self.ActionAboutThatProject.setShortcut("F1")
        self.ActionCV.setShortcut("Ctrl+V")
        self.ActionExport.setShortcut("Ctrl+E")
        self.ActionLoadModel.setShortcut("Ctrl+L")
        self.ActionModel.setShortcut("Ctrl+M")
        self.ActionNewModel.setShortcut("Ctrl+N")
        self.ActionPrediction.setShortcut("Ctrl+P")
        self.ActionQuit.setShortcut("Ctrl+Q")
        self.ActionSaveModel.setShortcut("Ctrl+S")

        self.MenuAbout.setTitle("&About")
        self.MenuChangeMode.setTitle("&Change mode")
        self.MenuOptions.setTitle("&Options")

        self.setupHandlers()
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

    def currentMode(self, value=None):
        """Both getter / setter for current mode"""
        if value is None:
            return self.__current_mode
        if isinstance(value, str):
            if value.lower() == 'model':
                self.__current_mode = 'model'
            elif value.lower() in ('crossvalidation', 'cv'):
                self.__current_mode = 'crossvalidation'
            elif value.lower() == 'prediction':
                self.__current_mode = 'prediction'
            elif value.lower() == 'start':
                self.__current_mode = 'start'
            else:
                IO.Log.error('Unknown mode ({}) passed to '
                             'currentMode()'.format(value))
                return
            self.CurrentModeLabel.setText(self.__current_mode.capitalize()
                                          + ' mode')
        else:
            IO.Log.error('currentMode() takes a string when used as a setter')

    def setupHandlers(self):
        self.ActionModel.triggered.connect(lambda: self.currentMode('model'))
        self.ActionCV.triggered.connect(lambda: self.currentMode('cv'))
        self.ActionPrediction.triggered.connect(lambda:
                                                self.currentMode('prediction'))

    def show(self):
        self.MainWindow.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    ui.show()
    sys.exit(app.exec_())
