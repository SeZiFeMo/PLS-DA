#!/usr/bin/env python
# coding: utf-8

import IO
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import sys


def set_size(widget, minimum=None, maximum=None, base=None):
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


def set_size_policy(widget, h_policy='Preferred', v_policy='Preferred',
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
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setObjectName("MainWindow")
        self.main_window.setEnabled(True)
        self.main_window.resize(800, 600)
        set_size_policy(self.main_window, 'Expanding', 'Expanding', 0, 0)
        set_size(self.main_window, minimum=(800, 600), maximum=(7680, 4320))
        self.main_window.setUnifiedTitleAndToolBarOnMac(True)
        self.setup()


    def retranslate(self):
        _translate = QtCore.QCoreApplication.translate
        self.main_window.setWindowTitle(_translate("MainWindow", "PLS-DA"))
        for index, entry in enumerate(self.drop_down_choices):
            self.LeftComboBox.setItemText(index, _translate("MainWindow", entry))

        self.LeftLVsLabel.setText(_translate("MainWindow", "Latent Variables"))
        self.LeftXRadioButton.setText(_translate("MainWindow", "&X"))
        self.LeftYRadioButton.setText(_translate("MainWindow", "&Y"))
        self.LeftPlotPushButton.setText(_translate("MainWindow", "Plot"))
        self.LeftBackPushButton.setText(_translate("MainWindow", "Back"))
        for index, entry in enumerate(self.drop_down_choices):
            self.CentralComboBox.setItemText(index, _translate("MainWindow", entry))

        self.CentralLVsLabel.setText(_translate("MainWindow", "Latent Variables"))
        self.CentralXRadioButton.setText(_translate("MainWindow", "&X"))
        self.CentralYRadioButton.setText(_translate("MainWindow", "&Y"))
        self.CentralBackPushButton.setText(_translate("MainWindow", "Back"))
        self.CentralPlotPushButton.setText(_translate("MainWindow", "Plot"))
        self.DetailsLabel.setText(_translate("MainWindow", "Details"))
        self.CurrentModeLabel.setText(_translate("MainWindow", "Current Mode"))
        self.MenuOptions.setTitle(_translate("MainWindow", "Opt&ions"))
        self.MenuChangeMode.setTitle(_translate("MainWindow", "&Change mode"))
        self.MenuAbout.setTitle(_translate("MainWindow", "&About"))
        self.ActionExport.setText(_translate("MainWindow", "&Export matrices"))
        self.ActionExport.setShortcut(_translate("MainWindow", "Ctrl+E"))
        self.ActionModel.setText(_translate("MainWindow", "&Model"))
        self.ActionModel.setShortcut(_translate("MainWindow", "Alt+M"))
        self.ActionCrossvalidation.setText(_translate("MainWindow", "&Crossvalidation"))
        self.ActionCrossvalidation.setShortcut(_translate("MainWindow", "Alt+V"))
        self.ActionPrediction.setText(_translate("MainWindow", "&Prediction"))
        self.ActionPrediction.setShortcut(_translate("MainWindow", "Alt+P"))
        self.ActionQuit.setText(_translate("MainWindow", "&Quit"))
        self.ActionQuit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.ActionSaveModel.setText(_translate("MainWindow", "&Save model"))
        self.ActionSaveModel.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.ActionLoadModel.setText(_translate("MainWindow", "&Load model"))
        self.ActionLoadModel.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.ActionNewModel.setText(_translate("MainWindow", "&New model"))
        self.ActionNewModel.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.ActionAboutThatProject.setText(_translate("MainWindow", "&About this project"))
        self.ActionAboutThatProject.setShortcut(_translate("MainWindow", "F1"))

    def setup(self):
        self.MainWidget = QtWidgets.QWidget(self.main_window)
        set_size_policy(self.MainWidget, 'Preferred', 'Preferred', 0, 0)
        set_size(self.MainWidget, minimum=(800, 600), maximum=(7680, 4300))
        self.MainWidget.setObjectName("MainWidget")

        self.MainSplitter = QtWidgets.QSplitter(self.MainWidget)
        set_size_policy(self.MainSplitter, 'Expanding', 'Expanding', 0, 0)
        set_size(self.MainSplitter, minimum=(800, 600), maximum=(7680, 4300))
        self.MainSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.MainSplitter.setHandleWidth(3)
        self.MainSplitter.setObjectName("MainSplitter")

        # Start creating widgets to put inside LeftWidget
        self.LeftScrollAreaWidgetContents = QtWidgets.QWidget()
        self.LeftScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 291, 565))
        set_size_policy(self.LeftScrollAreaWidgetContents, 'Expanding', 'Expanding', 0, 0)
        set_size(self.LeftScrollAreaWidgetContents, minimum=(174, 427), maximum=(3611, 4147))
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
        set_size(self.LeftLVsLabel, minimum=(70, 22), maximum=(1310, 170))
        self.LeftLVsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.LeftLVsLabel.setWordWrap(True)
        self.LeftLVsLabel.setObjectName("LeftLVsLabel")
        self.PlotFormLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.LeftLVsLabel)

        self.LeftLVsSpinBox = QtWidgets.QSpinBox(self.LeftScrollAreaWidgetContents)
        set_size_policy(self.LeftLVsSpinBox, 'Preferred', 'Preferred', 0, 0)
        set_size(self.LeftLVsSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftLVsSpinBox.setMinimum(1)
        self.LeftLVsSpinBox.setObjectName("LeftLVsSpinBox")
        self.PlotFormLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.LeftLVsSpinBox)

        self.LeftXRadioButton = QtWidgets.QRadioButton(self.LeftScrollAreaWidgetContents)
        set_size_policy(self.LeftXRadioButton, 'Preferred', 'Preferred', 0, 0)
        set_size(self.LeftXRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftXRadioButton.setObjectName("LeftXRadioButton")
        self.PlotFormLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.LeftXRadioButton)

        self.LeftYRadioButton = QtWidgets.QRadioButton(self.LeftScrollAreaWidgetContents)
        set_size_policy(self.LeftYRadioButton, 'Preferred', 'Preferred', 0, 0)
        set_size(self.LeftYRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftYRadioButton.setObjectName("LeftYRadioButton")
        self.PlotFormLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.LeftYRadioButton)

        self.LeftButtonGroup = QtWidgets.QButtonGroup(self.main_window)
        self.LeftButtonGroup.setObjectName("LeftButtonGroup")
        self.LeftButtonGroup.addButton(self.LeftXRadioButton)
        self.LeftButtonGroup.addButton(self.LeftYRadioButton)

        self.LeftXSpinBox = QtWidgets.QSpinBox(self.LeftScrollAreaWidgetContents)
        set_size_policy(self.LeftXSpinBox, 'Preferred', 'Preferred', 0, 0)
        set_size(self.LeftXSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftXSpinBox.setMinimum(1)
        self.LeftXSpinBox.setObjectName("LeftXSpinBox")
        self.PlotFormLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.LeftXSpinBox)

        self.LeftYSpinBox = QtWidgets.QSpinBox(self.LeftScrollAreaWidgetContents)
        set_size_policy(self.LeftYSpinBox, 'Preferred', 'Preferred', 0, 0)
        set_size(self.LeftYSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftYSpinBox.setMinimum(1)
        self.LeftYSpinBox.setObjectName("LeftYSpinBox")
        self.PlotFormLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.LeftYSpinBox)

        self.LeftPlotPushButton = QtWidgets.QPushButton(self.LeftScrollAreaWidgetContents)
        set_size_policy(self.LeftPlotPushButton, 'Preferred', 'Preferred', 0, 0)
        set_size(self.LeftPlotPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftPlotPushButton.setObjectName("LeftPlotPushButton")
        self.PlotFormLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.LeftPlotPushButton)

        self.LeftBackPushButton = QtWidgets.QPushButton(self.LeftScrollAreaWidgetContents)
        set_size_policy(self.LeftBackPushButton, 'Preferred', 'Preferred', 0, 0)
        set_size(self.LeftBackPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftBackPushButton.setObjectName("LeftBackPushButton")
        self.PlotFormLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.LeftBackPushButton)

        self.LeftWidget = QtWidgets.QWidget(self.MainSplitter)
        set_size(self.LeftWidget, minimum=(200, 580), maximum=(3637, 4300))
        self.LeftWidget.setObjectName("LeftWidget")

        self.LeftComboBox = QtWidgets.QComboBox(self.LeftWidget)
        set_size(self.LeftComboBox, minimum=(194, 22), maximum=(3631, 22))
        self.LeftComboBox.setObjectName("LeftComboBox")
        for entry in self.drop_down_choices:
            self.LeftComboBox.addItem("")

        self.LeftScrollArea = QtWidgets.QScrollArea(self.LeftWidget)
        set_size(self.LeftScrollArea, minimum=(194, 547), maximum=(3631, 4267))
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
        set_size_policy(self.CentralScrollAreaWidgetContents, 'Expanding', 'Expanding', 0, 0)
        set_size(self.CentralScrollAreaWidgetContents, minimum=(174, 427), maximum=(3611, 4147))
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
        set_size_policy(self.CentralLVsLabel, 'Preferred', 'Preferred', 0, 0)
        set_size(self.CentralLVsLabel, minimum=(70, 22), maximum=(1310, 170))
        self.CentralLVsLabel.setTextFormat(QtCore.Qt.AutoText)
        self.CentralLVsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.CentralLVsLabel.setWordWrap(True)
        self.CentralLVsLabel.setObjectName("CentralLVsLabel")
        self.PlotFormLayout1.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.CentralLVsLabel)

        self.CentralLVsSpinBox = QtWidgets.QSpinBox(self.CentralScrollAreaWidgetContents)
        set_size_policy(self.CentralLVsSpinBox, 'Preferred', 'Preferred', 0, 0)
        set_size(self.CentralLVsSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralLVsSpinBox.setMinimum(1)
        self.CentralLVsSpinBox.setObjectName("CentralLVsSpinBox")
        self.PlotFormLayout1.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.CentralLVsSpinBox)

        self.CentralXRadioButton = QtWidgets.QRadioButton(self.CentralScrollAreaWidgetContents)
        set_size_policy(self.CentralXRadioButton, 'Preferred', 'Preferred', 0, 0)
        set_size(self.CentralXRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.CentralXRadioButton.setObjectName("CentralXRadioButton")
        self.PlotFormLayout1.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.CentralXRadioButton)

        self.CentralYRadioButton = QtWidgets.QRadioButton(self.CentralScrollAreaWidgetContents)
        set_size_policy(self.CentralYRadioButton, 'Preferred', 'Preferred', 0, 0)
        set_size(self.CentralYRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.CentralYRadioButton.setObjectName("CentralYRadioButton")
        self.PlotFormLayout1.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.CentralYRadioButton)

        self.CentralButtonGroup = QtWidgets.QButtonGroup(self.main_window)
        self.CentralButtonGroup.setObjectName("CentralButtonGroup")
        self.CentralButtonGroup.addButton(self.CentralXRadioButton)
        self.CentralButtonGroup.addButton(self.CentralYRadioButton)

        self.CentralXSpinBox = QtWidgets.QSpinBox(self.CentralScrollAreaWidgetContents)
        set_size_policy(self.CentralXSpinBox, 'Preferred', 'Preferred', 0, 0)
        set_size(self.CentralXSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralXSpinBox.setMinimum(1)
        self.CentralXSpinBox.setObjectName("CentralXSpinBox")
        self.PlotFormLayout1.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.CentralXSpinBox)

        self.CentralYSpinBox = QtWidgets.QSpinBox(self.CentralScrollAreaWidgetContents)
        set_size_policy(self.CentralYSpinBox, 'Preferred', 'Preferred', 0, 0)
        set_size(self.CentralYSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralYSpinBox.setMinimum(1)
        self.CentralYSpinBox.setObjectName("CentralYSpinBox")
        self.PlotFormLayout1.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.CentralYSpinBox)

        self.CentralBackPushButton = QtWidgets.QPushButton(self.CentralScrollAreaWidgetContents)
        set_size_policy(self.CentralBackPushButton, 'Preferred', 'Preferred', 0, 0)
        set_size(self.CentralBackPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.CentralBackPushButton.setObjectName("CentralBackPushButton")
        self.PlotFormLayout1.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.CentralBackPushButton)

        self.CentralPlotPushButton = QtWidgets.QPushButton(self.CentralScrollAreaWidgetContents)
        set_size_policy(self.CentralPlotPushButton, 'Preferred', 'Preferred', 0, 0)
        set_size(self.CentralPlotPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.CentralPlotPushButton.setObjectName("CentralPlotPushButton")
        self.PlotFormLayout1.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.CentralPlotPushButton)

        self.CentralWidget = QtWidgets.QWidget(self.MainSplitter)
        set_size(self.CentralWidget, minimum=(200, 580), maximum=(3637, 4300))
        self.CentralWidget.setObjectName("CentralWidget")

        self.CentralComboBox = QtWidgets.QComboBox(self.CentralWidget)
        set_size_policy(self.CentralComboBox, 'Expanding', 'Expanding', 0, 0)
        set_size(self.CentralComboBox, minimum=(194, 22), maximum=(3631, 22))
        self.CentralComboBox.setObjectName("CentralComboBox")
        for entry in self.drop_down_choices:
            self.CentralComboBox.addItem("")

        self.CentralScrollArea = QtWidgets.QScrollArea(self.CentralWidget)
        set_size(self.CentralScrollArea, minimum=(194, 547), maximum=(3631, 4267))
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
        set_size_policy(self.RightScrollAreaWidgetContents, 'Expanding', 'Expanding', 0, 0)
        set_size(self.RightScrollAreaWidgetContents, minimum=(138, 534), maximum=(388, 4259))
        self.RightScrollAreaWidgetContents.setObjectName("RightScrollAreaWidgetContents")

        self.DetailsLabel = QtWidgets.QLabel(self.RightScrollAreaWidgetContents)
        set_size_policy(self.DetailsLabel, 'Expanding', 'Expanding', 0, 0)
        set_size(self.DetailsLabel, minimum=(138, 534), maximum=(388, 4259))
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
        set_size_policy(self.RightWidget, 'Expanding', 'Expanding', 0, 0)
        set_size(self.RightWidget, minimum=(150, 580), maximum=(400, 4300))
        self.RightWidget.setObjectName("RightWidget")

        self.RightScrollArea = QtWidgets.QScrollArea(self.RightWidget)
        set_size(self.RightScrollArea, minimum=(144, 547), maximum=(394, 4272))
        self.RightScrollArea.setWidgetResizable(True)
        self.RightScrollArea.setObjectName("RightScrollArea")
        self.RightScrollArea.setWidget(self.RightScrollAreaWidgetContents)

        self.CurrentModeLabel = QtWidgets.QLabel(self.RightWidget)
        set_size_policy(self.CurrentModeLabel, 'Expanding', 'Expanding', 0, 0)
        set_size(self.CurrentModeLabel, minimum=(144, 22), maximum=(394, 22))
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
        self.main_window.setCentralWidget(self.MainWidget)

        self.TopMenuBar = QtWidgets.QMenuBar(self.main_window)
        self.TopMenuBar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        set_size(self.TopMenuBar, minimum=(800, 20), maximum=(7680, 20))
        self.TopMenuBar.setObjectName("TopMenuBar")

        self.MenuOptions = QtWidgets.QMenu(self.TopMenuBar)
        set_size(self.MenuOptions, minimum=(100, 20), maximum=(960, 4300))
        self.MenuOptions.setObjectName("MenuOptions")

        self.MenuChangeMode = QtWidgets.QMenu(self.TopMenuBar)
        set_size(self.MenuChangeMode, minimum=(100, 20), maximum=(960, 4300))
        self.MenuChangeMode.setObjectName("MenuChangeMode")

        self.MenuAbout = QtWidgets.QMenu(self.TopMenuBar)
        set_size(self.MenuAbout, minimum=(100, 20), maximum=(960, 4300))
        self.MenuAbout.setObjectName("MenuAbout")

        self.main_window.setMenuBar(self.TopMenuBar)

        self.ActionExport = QtWidgets.QAction(self.main_window)
        self.ActionExport.setObjectName("ActionExport")

        self.ActionModel = QtWidgets.QAction(self.main_window)
        self.ActionModel.setObjectName("ActionModel")

        self.ActionCrossvalidation = QtWidgets.QAction(self.main_window)
        self.ActionCrossvalidation.setObjectName("ActionCrossvalidation")

        self.ActionPrediction = QtWidgets.QAction(self.main_window)
        self.ActionPrediction.setObjectName("ActionPrediction")

        self.ActionQuit = QtWidgets.QAction(self.main_window)
        self.ActionQuit.setObjectName("ActionQuit")

        self.ActionSaveModel = QtWidgets.QAction(self.main_window)
        self.ActionSaveModel.setObjectName("ActionSaveModel")

        self.ActionLoadModel = QtWidgets.QAction(self.main_window)
        self.ActionLoadModel.setObjectName("ActionLoadModel")

        self.ActionNewModel = QtWidgets.QAction(self.main_window)
        self.ActionNewModel.setObjectName("ActionNewModel")

        self.ActionAboutThatProject = QtWidgets.QAction(self.main_window)
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

        self.retranslate()
        QtCore.QMetaObject.connectSlotsByName(self.main_window)

    def show(self):
        self.main_window.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    ui.show()
    sys.exit(app.exec_())
