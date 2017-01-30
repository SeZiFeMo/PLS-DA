#!/usr/bin/env python
# coding: utf-8

import IO
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import sys


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
        self.MainWindow = QtWidgets.QMainWindow()
        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.setEnabled(True)
        self.MainWindow.resize(800, 600)
        setPolicy(self.MainWindow, 'Expanding', 'Expanding', 0, 0)
        setSize(self.MainWindow, minimum=(800, 600), maximum=(7680, 4320))
        self.MainWindow.setUnifiedTitleAndToolBarOnMac(True)

        # Previously in setupUi()
        self.MainWidget = QtWidgets.QWidget(self.MainWindow)
        setPolicy(self.MainWidget, 'Preferred', 'Preferred', 0, 0)
        setSize(self.MainWidget, minimum=(800, 600), maximum=(7680, 4300))
        self.MainWidget.setObjectName("MainWidget")

        self.MainSplitter = QtWidgets.QSplitter(self.MainWidget)
        setPolicy(self.MainSplitter, 'Expanding', 'Expanding', 0, 0)
        setSize(self.MainSplitter, minimum=(800, 600), maximum=(7680, 4300))
        self.MainSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.MainSplitter.setHandleWidth(3)
        self.MainSplitter.setObjectName("MainSplitter")

        # Start creating widgets to put inside LeftWidget
        self.LeftScrollAreaWidgetContents = QtWidgets.QWidget()
        self.LeftScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 291, 565))
        setPolicy(self.LeftScrollAreaWidgetContents, 'Expanding', 'Expanding', 0, 0)
        setSize(self.LeftScrollAreaWidgetContents, minimum=(174, 427), maximum=(3611, 4147))
        self.LeftScrollAreaWidgetContents.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.LeftScrollAreaWidgetContents.setObjectName("LeftScrollAreaWidgetContents")

        self.PlotFormLayout = QtWidgets.QFormLayout(self.LeftScrollAreaWidgetContents)
        self.PlotFormLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.PlotFormLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.PlotFormLayout.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.PlotFormLayout.setFormAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.PlotFormLayout.setContentsMargins(10, 10, 10, 10)
        self.PlotFormLayout.setSpacing(10)
        self.PlotFormLayout.setObjectName("PlotFormLayout")

        self.LeftLVsLabel = QtWidgets.QLabel(self.LeftScrollAreaWidgetContents)
        setSize(self.LeftLVsLabel, minimum=(70, 22), maximum=(1310, 170))
        self.LeftLVsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.LeftLVsLabel.setWordWrap(True)
        self.LeftLVsLabel.setObjectName("LeftLVsLabel")
        self.PlotFormLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.LeftLVsLabel)

        self.LeftLVsSpinBox = QtWidgets.QSpinBox(self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftLVsSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftLVsSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftLVsSpinBox.setMinimum(1)
        self.LeftLVsSpinBox.setObjectName("LeftLVsSpinBox")
        self.PlotFormLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.LeftLVsSpinBox)

        self.LeftXRadioButton = QtWidgets.QRadioButton(self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftXRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftXRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftXRadioButton.setObjectName("LeftXRadioButton")
        self.PlotFormLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.LeftXRadioButton)

        self.LeftYRadioButton = QtWidgets.QRadioButton(self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftYRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftYRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftYRadioButton.setObjectName("LeftYRadioButton")
        self.PlotFormLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.LeftYRadioButton)

        self.LeftButtonGroup = QtWidgets.QButtonGroup(self.MainWindow)
        self.LeftButtonGroup.setObjectName("LeftButtonGroup")
        self.LeftButtonGroup.addButton(self.LeftXRadioButton)
        self.LeftButtonGroup.addButton(self.LeftYRadioButton)

        self.LeftXSpinBox = QtWidgets.QSpinBox(self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftXSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftXSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftXSpinBox.setMinimum(1)
        self.LeftXSpinBox.setObjectName("LeftXSpinBox")
        self.PlotFormLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.LeftXSpinBox)

        self.LeftYSpinBox = QtWidgets.QSpinBox(self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftYSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftYSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftYSpinBox.setMinimum(1)
        self.LeftYSpinBox.setObjectName("LeftYSpinBox")
        self.PlotFormLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.LeftYSpinBox)

        self.LeftPlotPushButton = QtWidgets.QPushButton(self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftPlotPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftPlotPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftPlotPushButton.setObjectName("LeftPlotPushButton")
        self.PlotFormLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.LeftPlotPushButton)

        self.LeftBackPushButton = QtWidgets.QPushButton(self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftBackPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftBackPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftBackPushButton.setObjectName("LeftBackPushButton")
        self.PlotFormLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.LeftBackPushButton)

        self.LeftWidget = QtWidgets.QWidget(self.MainSplitter)
        setSize(self.LeftWidget, minimum=(200, 580), maximum=(3637, 4300))
        self.LeftWidget.setObjectName("LeftWidget")

        self.LeftComboBox = QtWidgets.QComboBox(self.LeftWidget)
        setSize(self.LeftComboBox, minimum=(194, 22), maximum=(3631, 22))
        self.LeftComboBox.setObjectName("LeftComboBox")
        for entry in self.drop_down_choices:
            self.LeftComboBox.addItem("")

        self.LeftScrollArea = QtWidgets.QScrollArea(self.LeftWidget)
        setSize(self.LeftScrollArea, minimum=(194, 547), maximum=(3631, 4267))
        self.LeftScrollArea.setWidgetResizable(True)
        self.LeftScrollArea.setObjectName("LeftScrollArea")
        self.LeftScrollArea.setWidget(self.LeftScrollAreaWidgetContents)

        self.LeftGridLayout = QtWidgets.QGridLayout(self.LeftWidget)
        self.LeftGridLayout.setContentsMargins(3, 3, 3, 3)
        self.LeftGridLayout.setSpacing(5)
        self.LeftGridLayout.setObjectName("LeftGridLayout")
        self.LeftGridLayout.addWidget(self.LeftComboBox, 0, 0, 1, 1)
        self.LeftGridLayout.addWidget(self.LeftScrollArea, 1, 0, 1, 1)

        # Start creating widgets to put inside CentralWidget
        self.CentralScrollAreaWidgetContents = QtWidgets.QWidget()
        self.CentralScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 290, 565))
        setPolicy(self.CentralScrollAreaWidgetContents, 'Expanding', 'Expanding', 0, 0)
        setSize(self.CentralScrollAreaWidgetContents, minimum=(174, 427), maximum=(3611, 4147))
        self.CentralScrollAreaWidgetContents.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.CentralScrollAreaWidgetContents.setObjectName("CentralScrollAreaWidgetContents")

        self.PlotFormLayout1 = QtWidgets.QFormLayout(self.CentralScrollAreaWidgetContents)
        self.PlotFormLayout1.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.PlotFormLayout1.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.PlotFormLayout1.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.PlotFormLayout1.setFormAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.PlotFormLayout1.setContentsMargins(10, 10, 10, 10)
        self.PlotFormLayout1.setSpacing(10)
        self.PlotFormLayout1.setObjectName("PlotFormLayout1")

        self.CentralLVsLabel = QtWidgets.QLabel(self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralLVsLabel, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralLVsLabel, minimum=(70, 22), maximum=(1310, 170))
        self.CentralLVsLabel.setTextFormat(QtCore.Qt.AutoText)
        self.CentralLVsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.CentralLVsLabel.setWordWrap(True)
        self.CentralLVsLabel.setObjectName("CentralLVsLabel")
        self.PlotFormLayout1.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.CentralLVsLabel)

        self.CentralLVsSpinBox = QtWidgets.QSpinBox(self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralLVsSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralLVsSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralLVsSpinBox.setMinimum(1)
        self.CentralLVsSpinBox.setObjectName("CentralLVsSpinBox")
        self.PlotFormLayout1.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.CentralLVsSpinBox)

        self.CentralXRadioButton = QtWidgets.QRadioButton(self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralXRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralXRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.CentralXRadioButton.setObjectName("CentralXRadioButton")
        self.PlotFormLayout1.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.CentralXRadioButton)

        self.CentralYRadioButton = QtWidgets.QRadioButton(self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralYRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralYRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.CentralYRadioButton.setObjectName("CentralYRadioButton")
        self.PlotFormLayout1.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.CentralYRadioButton)

        self.CentralButtonGroup = QtWidgets.QButtonGroup(self.MainWindow)
        self.CentralButtonGroup.setObjectName("CentralButtonGroup")
        self.CentralButtonGroup.addButton(self.CentralXRadioButton)
        self.CentralButtonGroup.addButton(self.CentralYRadioButton)

        self.CentralXSpinBox = QtWidgets.QSpinBox(self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralXSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralXSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralXSpinBox.setMinimum(1)
        self.CentralXSpinBox.setObjectName("CentralXSpinBox")
        self.PlotFormLayout1.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.CentralXSpinBox)

        self.CentralYSpinBox = QtWidgets.QSpinBox(self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralYSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralYSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralYSpinBox.setMinimum(1)
        self.CentralYSpinBox.setObjectName("CentralYSpinBox")
        self.PlotFormLayout1.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.CentralYSpinBox)

        self.CentralBackPushButton = QtWidgets.QPushButton(self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralBackPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralBackPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.CentralBackPushButton.setObjectName("CentralBackPushButton")
        self.PlotFormLayout1.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.CentralBackPushButton)

        self.CentralPlotPushButton = QtWidgets.QPushButton(self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralPlotPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralPlotPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.CentralPlotPushButton.setObjectName("CentralPlotPushButton")
        self.PlotFormLayout1.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.CentralPlotPushButton)

        self.CentralWidget = QtWidgets.QWidget(self.MainSplitter)
        setSize(self.CentralWidget, minimum=(200, 580), maximum=(3637, 4300))
        self.CentralWidget.setObjectName("CentralWidget")

        self.CentralComboBox = QtWidgets.QComboBox(self.CentralWidget)
        setPolicy(self.CentralComboBox, 'Expanding', 'Expanding', 0, 0)
        setSize(self.CentralComboBox, minimum=(194, 22), maximum=(3631, 22))
        self.CentralComboBox.setObjectName("CentralComboBox")
        for entry in self.drop_down_choices:
            self.CentralComboBox.addItem("")

        self.CentralScrollArea = QtWidgets.QScrollArea(self.CentralWidget)
        setSize(self.CentralScrollArea, minimum=(194, 547), maximum=(3631, 4267))
        self.CentralScrollArea.setWidgetResizable(True)
        self.CentralScrollArea.setObjectName("CentralScrollArea")
        self.CentralScrollArea.setWidget(self.CentralScrollAreaWidgetContents)

        self.CentralGridLayout = QtWidgets.QGridLayout(self.CentralWidget)
        self.CentralGridLayout.setContentsMargins(3, 3, 3, 3)
        self.CentralGridLayout.setSpacing(5)
        self.CentralGridLayout.setObjectName("CentralGridLayout")
        self.CentralGridLayout.addWidget(self.CentralComboBox, 0, 0, 1, 1)
        self.CentralGridLayout.addWidget(self.CentralScrollArea, 1, 0, 1, 1)

        # Start creating widgets to put inside RightWidget
        self.RightScrollAreaWidgetContents = QtWidgets.QWidget()
        self.RightScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 189, 565))
        setPolicy(self.RightScrollAreaWidgetContents, 'Expanding', 'Expanding', 0, 0)
        setSize(self.RightScrollAreaWidgetContents, minimum=(138, 534), maximum=(388, 4259))
        self.RightScrollAreaWidgetContents.setObjectName("RightScrollAreaWidgetContents")

        self.DetailsLabel = QtWidgets.QLabel(self.RightScrollAreaWidgetContents)
        setPolicy(self.DetailsLabel, 'Expanding', 'Expanding', 0, 0)
        setSize(self.DetailsLabel, minimum=(138, 534), maximum=(388, 4259))
        self.DetailsLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.DetailsLabel.setWordWrap(True)
        self.DetailsLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.DetailsLabel.setObjectName("DetailsLabel")

        self.gridLayout = QtWidgets.QGridLayout(self.RightScrollAreaWidgetContents)
        self.gridLayout.setContentsMargins(3, 3, 3, 3)
        self.gridLayout.setSpacing(5)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.addWidget(self.DetailsLabel, 0, 0, 1, 1)

        self.RightWidget = QtWidgets.QWidget(self.MainSplitter)
        setPolicy(self.RightWidget, 'Expanding', 'Expanding', 0, 0)
        setSize(self.RightWidget, minimum=(150, 580), maximum=(400, 4300))
        self.RightWidget.setObjectName("RightWidget")

        self.RightScrollArea = QtWidgets.QScrollArea(self.RightWidget)
        setSize(self.RightScrollArea, minimum=(144, 547), maximum=(394, 4272))
        self.RightScrollArea.setWidgetResizable(True)
        self.RightScrollArea.setObjectName("RightScrollArea")
        self.RightScrollArea.setWidget(self.RightScrollAreaWidgetContents)

        self.CurrentModeLabel = QtWidgets.QLabel(self.RightWidget)
        setPolicy(self.CurrentModeLabel, 'Expanding', 'Expanding', 0, 0)
        setSize(self.CurrentModeLabel, minimum=(144, 22), maximum=(394, 22))
        self.CurrentModeLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.CurrentModeLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.CurrentModeLabel.setLineWidth(1)
        self.CurrentModeLabel.setTextFormat(QtCore.Qt.AutoText)
        self.CurrentModeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.CurrentModeLabel.setObjectName("CurrentModeLabel")

        self.RightGridLayout = QtWidgets.QGridLayout(self.RightWidget)
        self.RightGridLayout.setContentsMargins(3, 3, 3, 3)
        self.RightGridLayout.setSpacing(5)
        self.RightGridLayout.setObjectName("RightGridLayout")
        self.RightGridLayout.addWidget(self.RightScrollArea, 1, 0, 1, 1)
        self.RightGridLayout.addWidget(self.CurrentModeLabel, 0, 0, 1, 1)

        self.MainGridLayout = QtWidgets.QGridLayout(self.MainWidget)
        self.MainGridLayout.setContentsMargins(0, 0, 0, 0)
        self.MainGridLayout.setSpacing(0)
        self.MainGridLayout.setObjectName("MainGridLayout")
        self.MainGridLayout.addWidget(self.MainSplitter, 0, 0, 1, 1)
        self.MainWindow.setCentralWidget(self.MainWidget)

        self.TopMenuBar = QtWidgets.QMenuBar(self.MainWindow)
        self.TopMenuBar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        setSize(self.TopMenuBar, minimum=(800, 20), maximum=(7680, 20))
        self.TopMenuBar.setObjectName("TopMenuBar")

        self.MenuOptions = QtWidgets.QMenu(self.TopMenuBar)
        setSize(self.MenuOptions, minimum=(100, 20), maximum=(960, 4300))
        self.MenuOptions.setObjectName("MenuOptions")

        self.MenuChangeMode = QtWidgets.QMenu(self.TopMenuBar)
        setSize(self.MenuChangeMode, minimum=(100, 20), maximum=(960, 4300))
        self.MenuChangeMode.setObjectName("MenuChangeMode")

        self.MenuAbout = QtWidgets.QMenu(self.TopMenuBar)
        setSize(self.MenuAbout, minimum=(100, 20), maximum=(960, 4300))
        self.MenuAbout.setObjectName("MenuAbout")

        self.MainWindow.setMenuBar(self.TopMenuBar)

        self.ActionExport = QtWidgets.QAction(self.MainWindow)
        self.ActionExport.setObjectName("ActionExport")

        self.ActionModel = QtWidgets.QAction(self.MainWindow)
        self.ActionModel.setObjectName("ActionModel")

        self.ActionCrossvalidation = QtWidgets.QAction(self.MainWindow)
        self.ActionCrossvalidation.setObjectName("ActionCrossvalidation")

        self.ActionPrediction = QtWidgets.QAction(self.MainWindow)
        self.ActionPrediction.setObjectName("ActionPrediction")

        self.ActionQuit = QtWidgets.QAction(self.MainWindow)
        self.ActionQuit.setObjectName("ActionQuit")

        self.ActionSaveModel = QtWidgets.QAction(self.MainWindow)
        self.ActionSaveModel.setObjectName("ActionSaveModel")

        self.ActionLoadModel = QtWidgets.QAction(self.MainWindow)
        self.ActionLoadModel.setObjectName("ActionLoadModel")

        self.ActionNewModel = QtWidgets.QAction(self.MainWindow)
        self.ActionNewModel.setObjectName("ActionNewModel")

        self.ActionAboutThatProject = QtWidgets.QAction(self.MainWindow)
        self.ActionAboutThatProject.setObjectName("ActionAboutThatProject")

        self.MenuOptions.addAction(self.ActionNewModel)
        self.MenuOptions.addAction(self.ActionSaveModel)
        self.MenuOptions.addAction(self.ActionLoadModel)
        self.MenuOptions.addSeparator()
        self.MenuOptions.addAction(self.ActionExport)
        self.MenuOptions.addSeparator()
        self.MenuOptions.addAction(self.ActionQuit)

        self.MenuChangeMode.addAction(self.ActionModel)
        self.MenuChangeMode.addAction(self.ActionCrossvalidation)
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
        self.CurrentModeLabel.setText("Current Mode")

        self.ActionAboutThatProject.setText("A&bout this project")
        self.ActionCrossvalidation.setText("Cross&Validation")
        self.ActionExport.setText("&Export matrices")
        self.ActionLoadModel.setText("&Load model")
        self.ActionModel.setText("&Model")
        self.ActionNewModel.setText("&New model")
        self.ActionPrediction.setText("&Prediction")
        self.ActionQuit.setText("&Quit")
        self.ActionSaveModel.setText("&Save model")

        self.ActionAboutThatProject.setShortcut("F1")
        self.ActionCrossvalidation.setShortcut("Ctrl+V")
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

        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

    def show(self):
        self.MainWindow.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    ui.show()
    sys.exit(app.exec_())
