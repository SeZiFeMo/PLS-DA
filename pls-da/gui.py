#!/usr/bin/env python3
# coding: utf-8

import model
import os
import pickle
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import sys

import IO
import utility


def new_qt(widget, name, parent=None):
    """Return a widget of type specified with 1st argument (a string).

       Also set the object name and optionally the parent widjet.
    """
    ret = getattr(QtWidgets, widget)(parent)
    ret.setObjectName(name)
    return ret


def popupChooseFile(parent, input=False, output=False, name_filter=None):
    """Display a dialog to choose a file.

       Raises Exception if input and outpu flag does not differ.
       Return file path or None.
    """
    if bool(input) == bool(output):
        raise Exception('In popupChooseFile() i/o flags must differ')
    mode = 'input' if input else 'output'

    IO.Log.debug('Choose an {} file'.format(mode))
    dialog = new_qt('QFileDialog', 'popupChooseFile', parent=parent)
    dialog.setWindowTitle('Choose an {} file'.format(mode))
    if input:
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
    else:
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    if name_filter is not None:
        dialog.setNameFilter(name_filter)
    if not dialog.exec():
        IO.Log.debug('CANCEL (not chosen {} file)'.format(mode))
        return None
    IO.Log.debug('OK (chosen {} file)'.format(mode), dialog.selectedFiles()[0])
    return dialog.selectedFiles()[0]


def popupChooseItem(message, item_list, parent, title=None):
    """Display a dialog to choose an item from a list.

       Return tuple with two values:
         True on Ok answer, False otherwise
         index of the chosen item in item_list.
    """
    dialog = new_qt('QInputDialog', 'popupChooseItem', parent=parent)
    if title is not None:
        dialog.setWindowTitle(title)
    dialog.setLabelText(message)
    dialog.setInputMode(QtWidgets.QInputDialog.TextInput)
    dialog.setComboBoxItems(item_list)
    ok_answer = dialog.exec() == QtWidgets.QDialog.Accepted
    return ok_answer, item_list.index(dialog.textValue())


def popupError(message, parent):
    """Display a dialog with an informative message and an Ok button."""
    dialog = new_qt('QMessageBox', 'popupError', parent=parent)
    dialog.setIcon(QtWidgets.QMessageBox.Critical)
    if not isinstance(message, str):
        message = str(message)
    dialog.setText(message)
    dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
    dialog.exec()


def popupQuestion(message, parent, title=None):
    """Display a dialog with a question.

       Return True on Yes answer, False otherwise.
    """
    dialog = new_qt('QMessageBox', 'popupQuestion', parent=parent)
    if title is not None:
        dialog.setWindowTitle(title)
    dialog.setText(message)
    dialog.setStandardButtons(QtWidgets.QMessageBox.Yes
                              | QtWidgets.QMessageBox.No)
    choice = dialog.exec()
    return choice == QtWidgets.QMessageBox.Yes


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

    csv_file_filter = "Comma-separated values files (*.csv *.txt)"
    pickle_file_filter = "Python pickle files (*.p *.pkl *.pickle)"

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

        self.MainSplitter = new_qt('QSplitter', 'MainSplitter',
                                parent=self.MainWidget)
        setPolicy(self.MainSplitter, 'Expanding', 'Expanding', 0, 0)
        setSize(self.MainSplitter, minimum=(800, 600), maximum=(7680, 4300))
        self.MainSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.MainSplitter.setHandleWidth(3)

        # Start creating widgets to put inside LeftWidget
        self.LeftScrollAreaWidgetContents = new_qt('QWidget',
                                                'LeftScrollAreaWidgetContents')
        self.LeftScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0,
                                                                   291, 565))
        setPolicy(self.LeftScrollAreaWidgetContents,
                  'Expanding', 'Expanding', 0, 0)
        setSize(self.LeftScrollAreaWidgetContents,
                minimum=(174, 427), maximum=(3611, 4147))
        self.LeftScrollAreaWidgetContents.setLayoutDirection(
                QtCore.Qt.LeftToRight)

        self.LeftPlotFormLayout = new_qt('QFormLayout', 'LeftPlotFormLayout',
                                      parent=self.LeftScrollAreaWidgetContents)
        self.LeftPlotFormLayout.setSizeConstraint(
                QtWidgets.QLayout.SetMaximumSize)
        self.LeftPlotFormLayout.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.LeftPlotFormLayout.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.LeftPlotFormLayout.setFormAlignment(QtCore.Qt.AlignHCenter
                                                 | QtCore.Qt.AlignTop)
        self.LeftPlotFormLayout.setContentsMargins(10, 10, 10, 10)
        self.LeftPlotFormLayout.setSpacing(10)

        self.LeftLVsLabel = new_qt('QLabel', 'LeftLVsLabel',
                                parent=self.LeftScrollAreaWidgetContents)
        setSize(self.LeftLVsLabel, minimum=(70, 22), maximum=(1310, 170))
        self.LeftLVsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.LeftLVsLabel.setWordWrap(True)
        self.LeftPlotFormLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole,
                                          self.LeftLVsLabel)

        self.LeftLVsSpinBox = new_qt('QSpinBox', 'LeftLVsSpinBox',
                                  parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftLVsSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftLVsSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftLVsSpinBox.setMinimum(1)
        self.LeftPlotFormLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole,
                                          self.LeftLVsSpinBox)

        self.LeftXRadioButton = new_qt('QRadioButton', 'LeftXRadioButton',
                                    parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftXRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftXRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftPlotFormLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole,
                                          self.LeftXRadioButton)

        self.LeftYRadioButton = new_qt('QRadioButton', 'LeftYRadioButton',
                                    parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftYRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftYRadioButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftPlotFormLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole,
                                          self.LeftYRadioButton)

        self.LeftButtonGroup = new_qt('QButtonGroup', 'LeftButtonGroup',
                                   parent=self.LeftScrollAreaWidgetContents)
        self.LeftButtonGroup.addButton(self.LeftXRadioButton)
        self.LeftButtonGroup.addButton(self.LeftYRadioButton)

        self.LeftXSpinBox = new_qt('QSpinBox', 'LeftXSpinBox',
                                parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftXSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftXSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftXSpinBox.setMinimum(1)
        self.LeftPlotFormLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole,
                                          self.LeftXSpinBox)

        self.LeftYSpinBox = new_qt('QSpinBox', 'LeftYSpinBox',
                                parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftYSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftYSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.LeftYSpinBox.setMinimum(1)
        self.LeftPlotFormLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole,
                                          self.LeftYSpinBox)

        self.LeftPlotPushButton = new_qt('QPushButton', 'LeftPlotPushButton',
                                      parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftPlotPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftPlotPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftPlotFormLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole,
                                          self.LeftPlotPushButton)

        self.LeftBackPushButton = new_qt('QPushButton', 'LeftBackPushButton',
                                      parent=self.LeftScrollAreaWidgetContents)
        setPolicy(self.LeftBackPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.LeftBackPushButton, minimum=(70, 22), maximum=(1310, 170))
        self.LeftPlotFormLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole,
                                          self.LeftBackPushButton)

        self.LeftWidget = new_qt('QWidget', 'LeftWidget',
                              parent=self.MainSplitter)
        setSize(self.LeftWidget, minimum=(200, 580), maximum=(3637, 4300))

        self.LeftComboBox = new_qt('QComboBox', 'LeftComboBox',
                                parent=self.LeftWidget)
        setSize(self.LeftComboBox, minimum=(194, 22), maximum=(3631, 22))
        for entry in self.drop_down_choices:
            self.LeftComboBox.addItem("")

        self.LeftScrollArea = new_qt('QScrollArea', 'LeftScrollArea',
                                  parent=self.LeftWidget)
        setSize(self.LeftScrollArea, minimum=(194, 547), maximum=(3631, 4267))
        self.LeftScrollArea.setWidgetResizable(True)
        self.LeftScrollArea.setWidget(self.LeftScrollAreaWidgetContents)

        self.LeftGridLayout = new_qt('QGridLayout', 'LeftGridLayout',
                                  parent=self.LeftWidget)
        self.LeftGridLayout.setContentsMargins(3, 3, 3, 3)
        self.LeftGridLayout.setSpacing(5)
        self.LeftGridLayout.addWidget(self.LeftComboBox, 0, 0, 1, 1)
        self.LeftGridLayout.addWidget(self.LeftScrollArea, 1, 0, 1, 1)

        # Start creating widgets to put inside CentralWidget
        self.CentralScrollAreaWidgetContents = new_qt(
                'QWidget', 'CentralScrollAreaWidgetContents')
        self.CentralScrollAreaWidgetContents.setGeometry(
                QtCore.QRect(0, 0, 290, 565))
        setPolicy(self.CentralScrollAreaWidgetContents,
                  'Expanding', 'Expanding', 0, 0)
        setSize(self.CentralScrollAreaWidgetContents,
                minimum=(174, 427), maximum=(3611, 4147))
        self.CentralScrollAreaWidgetContents.setLayoutDirection(
                QtCore.Qt.LeftToRight)

        self.CentralPlotFormLayout = new_qt(
                'QFormLayout', 'CentralPlotFormLayout',
                parent=self.CentralScrollAreaWidgetContents)
        self.CentralPlotFormLayout.setSizeConstraint(
                QtWidgets.QLayout.SetMaximumSize)
        self.CentralPlotFormLayout.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.CentralPlotFormLayout.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.CentralPlotFormLayout.setFormAlignment(QtCore.Qt.AlignHCenter
                                                    | QtCore.Qt.AlignTop)
        self.CentralPlotFormLayout.setContentsMargins(10, 10, 10, 10)
        self.CentralPlotFormLayout.setSpacing(10)

        self.CentralLVsLabel = new_qt('QLabel', 'CentralLVsLabel',
                                   parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralLVsLabel, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralLVsLabel, minimum=(70, 22), maximum=(1310, 170))
        self.CentralLVsLabel.setTextFormat(QtCore.Qt.AutoText)
        self.CentralLVsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.CentralLVsLabel.setWordWrap(True)
        self.CentralPlotFormLayout.setWidget(0,
                                             QtWidgets.QFormLayout.LabelRole,
                                             self.CentralLVsLabel)

        self.CentralLVsSpinBox = new_qt(
                'QSpinBox', 'CentralLVsSpinBox',
                parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralLVsSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralLVsSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralLVsSpinBox.setMinimum(1)
        self.CentralPlotFormLayout.setWidget(0,
                                             QtWidgets.QFormLayout.FieldRole,
                                             self.CentralLVsSpinBox)

        self.CentralXRadioButton = new_qt(
                'QRadioButton', 'CentralXRadioButton',
                parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralXRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralXRadioButton,
                minimum=(70, 22), maximum=(1310, 170))
        self.CentralPlotFormLayout.setWidget(1,
                                             QtWidgets.QFormLayout.LabelRole,
                                             self.CentralXRadioButton)

        self.CentralYRadioButton = new_qt(
                'QRadioButton', 'CentralYRadioButton',
                parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralYRadioButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralYRadioButton,
                minimum=(70, 22), maximum=(1310, 170))
        self.CentralPlotFormLayout.setWidget(1,
                                             QtWidgets.QFormLayout.FieldRole,
                                             self.CentralYRadioButton)

        self.CentralButtonGroup = new_qt(
                'QButtonGroup', 'CentralButtonGroup',
                parent=self.CentralScrollAreaWidgetContents)
        self.CentralButtonGroup.addButton(self.CentralXRadioButton)
        self.CentralButtonGroup.addButton(self.CentralYRadioButton)

        self.CentralXSpinBox = new_qt(
                'QSpinBox', 'CentralXSpinBox',
                parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralXSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralXSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralXSpinBox.setMinimum(1)
        self.CentralPlotFormLayout.setWidget(2,
                                             QtWidgets.QFormLayout.LabelRole,
                                             self.CentralXSpinBox)

        self.CentralYSpinBox = new_qt('QSpinBox', 'CentralYSpinBox',
                                   parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralYSpinBox, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralYSpinBox, minimum=(70, 22), maximum=(1310, 170))
        self.CentralYSpinBox.setMinimum(1)
        self.CentralPlotFormLayout.setWidget(2,
                                             QtWidgets.QFormLayout.FieldRole,
                                             self.CentralYSpinBox)

        self.CentralBackPushButton = new_qt(
                'QPushButton', 'CentralBackPushButton',
                parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralBackPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralBackPushButton,
                minimum=(70, 22), maximum=(1310, 170))
        self.CentralPlotFormLayout.setWidget(3,
                                             QtWidgets.QFormLayout.LabelRole,
                                             self.CentralBackPushButton)

        self.CentralPlotPushButton = new_qt(
                'QPushButton', 'CentralPlotPushButton',
                parent=self.CentralScrollAreaWidgetContents)
        setPolicy(self.CentralPlotPushButton, 'Preferred', 'Preferred', 0, 0)
        setSize(self.CentralPlotPushButton,
                minimum=(70, 22), maximum=(1310, 170))
        self.CentralPlotFormLayout.setWidget(3,
                                             QtWidgets.QFormLayout.FieldRole,
                                             self.CentralPlotPushButton)

        self.CentralWidget = new_qt('QWidget', 'CentralWidget',
                                 parent=self.MainSplitter)
        setSize(self.CentralWidget, minimum=(200, 580), maximum=(3637, 4300))

        self.CentralComboBox = new_qt('QComboBox', 'CentralComboBox',
                                   parent=self.CentralWidget)
        setPolicy(self.CentralComboBox, 'Expanding', 'Expanding', 0, 0)
        setSize(self.CentralComboBox, minimum=(194, 22), maximum=(3631, 22))
        for entry in self.drop_down_choices:
            self.CentralComboBox.addItem("")

        self.CentralScrollArea = new_qt('QScrollArea', 'CentralScrollArea',
                                     parent=self.CentralWidget)
        setSize(self.CentralScrollArea,
                minimum=(194, 547), maximum=(3631, 4267))
        self.CentralScrollArea.setWidgetResizable(True)
        self.CentralScrollArea.setWidget(self.CentralScrollAreaWidgetContents)

        self.CentralGridLayout = new_qt('QGridLayout', 'CentralGridLayout',
                                     parent=self.CentralWidget)
        self.CentralGridLayout.setContentsMargins(3, 3, 3, 3)
        self.CentralGridLayout.setSpacing(5)
        self.CentralGridLayout.addWidget(self.CentralComboBox, 0, 0, 1, 1)
        self.CentralGridLayout.addWidget(self.CentralScrollArea, 1, 0, 1, 1)

        # Start creating widgets to put inside RightWidget
        self.RightScrollAreaWidgetContents = new_qt(
                'QWidget', 'RightScrollAreaWidgetContents')
        self.RightScrollAreaWidgetContents.setGeometry(
                QtCore.QRect(0, 0, 189, 565))
        setPolicy(self.RightScrollAreaWidgetContents,
                  'Expanding', 'Expanding', 0, 0)
        setSize(self.RightScrollAreaWidgetContents,
                minimum=(138, 534), maximum=(388, 4259))

        self.DetailsLabel = new_qt('QLabel', 'DetailsLabel',
                                parent=self.RightScrollAreaWidgetContents)
        setPolicy(self.DetailsLabel, 'Expanding', 'Expanding', 0, 0)
        setSize(self.DetailsLabel, minimum=(138, 534), maximum=(388, 4259))
        self.DetailsLabel.setAlignment(QtCore.Qt.AlignHCenter
                                       | QtCore.Qt.AlignTop)
        self.DetailsLabel.setWordWrap(True)
        self.DetailsLabel.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByKeyboard
                | QtCore.Qt.TextSelectableByMouse)

        self.gridLayout = new_qt('QGridLayout', 'gridLayout',
                              parent=self.RightScrollAreaWidgetContents)
        self.gridLayout.setContentsMargins(3, 3, 3, 3)
        self.gridLayout.setSpacing(5)
        self.gridLayout.addWidget(self.DetailsLabel, 0, 0, 1, 1)

        self.RightWidget = new_qt('QWidget', 'RightWidget',
                               parent=self.MainSplitter)
        setPolicy(self.RightWidget, 'Expanding', 'Expanding', 0, 0)
        setSize(self.RightWidget, minimum=(150, 580), maximum=(400, 4300))

        self.RightScrollArea = new_qt('QScrollArea', 'RightScrollArea',
                                   parent=self.RightWidget)
        setSize(self.RightScrollArea, minimum=(144, 547), maximum=(394, 4272))
        self.RightScrollArea.setWidgetResizable(True)
        self.RightScrollArea.setWidget(self.RightScrollAreaWidgetContents)

        self.CurrentModeLabel = new_qt('QLabel', 'CurrentModeLabel',
                                    parent=self.RightWidget)
        setPolicy(self.CurrentModeLabel, 'Expanding', 'Expanding', 0, 0)
        setSize(self.CurrentModeLabel, minimum=(144, 22), maximum=(394, 22))
        self.CurrentModeLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.CurrentModeLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.CurrentModeLabel.setLineWidth(1)
        self.CurrentModeLabel.setTextFormat(QtCore.Qt.AutoText)
        self.CurrentModeLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.RightGridLayout = new_qt('QGridLayout', 'RightGridLayout',
                                   parent=self.RightWidget)
        self.RightGridLayout.setContentsMargins(3, 3, 3, 3)
        self.RightGridLayout.setSpacing(5)
        self.RightGridLayout.addWidget(self.RightScrollArea, 1, 0, 1, 1)
        self.RightGridLayout.addWidget(self.CurrentModeLabel, 0, 0, 1, 1)

        self.MainGridLayout = new_qt('QGridLayout', 'MainGridLayout',
                                  parent=self.MainWidget)
        self.MainGridLayout.setContentsMargins(0, 0, 0, 0)
        self.MainGridLayout.setSpacing(0)
        self.MainGridLayout.addWidget(self.MainSplitter, 0, 0, 1, 1)
        self.MainWindow.setCentralWidget(self.MainWidget)

        self.TopMenuBar = new_qt('QMenuBar', 'TopMenuBar', parent=self.MainWindow)
        self.TopMenuBar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        setSize(self.TopMenuBar, minimum=(800, 20), maximum=(7680, 20))

        self.MenuOptions = new_qt('QMenu', 'MenuOptions', parent=self.TopMenuBar)
        setSize(self.MenuOptions, minimum=(100, 20), maximum=(960, 4300))

        self.MenuChangeMode = new_qt('QMenu', 'MenuChangeMode',
                                  parent=self.TopMenuBar)
        setSize(self.MenuChangeMode, minimum=(100, 20), maximum=(960, 4300))

        self.MenuAbout = new_qt('QMenu', 'MenuAbout', parent=self.TopMenuBar)
        setSize(self.MenuAbout, minimum=(100, 20), maximum=(960, 4300))

        self.MainWindow.setMenuBar(self.TopMenuBar)

        self.ActionNewModel = new_qt('QAction', 'ActionNewModel',
                                  parent=self.MenuOptions)
        self.ActionSaveModel = new_qt('QAction', 'ActionSaveModel',
                                   parent=self.MenuOptions)
        self.ActionLoadModel = new_qt('QAction', 'ActionLoadModel',
                                   parent=self.MenuOptions)
        self.ActionExport = new_qt('QAction', 'ActionExport',
                                parent=self.MenuOptions)
        self.ActionQuit = new_qt('QAction', 'ActionQuit', parent=self.MenuOptions)

        self.ActionModel = new_qt('QAction', 'ActionModel',
                               parent=self.MenuChangeMode)
        self.ActionCV = new_qt('QAction', 'ActionCV', parent=self.MenuChangeMode)
        self.ActionPrediction = new_qt('QAction', 'ActionPrediction',
                                    parent=self.MenuChangeMode)

        self.ActionAboutThatProject = new_qt('QAction', 'ActionAboutThatProject',
                                          parent=self.MenuAbout)

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

        self.MenuAbout.setTitle("&About")
        self.MenuChangeMode.setTitle("&Change mode")
        self.MenuOptions.setTitle("&Options")

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
        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

        self.setupHandlers()
        self.setupStatusAttributes()

    def currentMode(self, value=None):
        """Both getter and setter of current mode."""
        if value is None:
            return self.__current_mode
        if isinstance(value, str):
            if value.lower() == 'model':
                self.__current_mode = 'model'
                IO.Log.debug('Current mode changed into "MODEL"')
            elif value.lower() in ('crossvalidation', 'cv'):
                self.__current_mode = 'crossvalidation'
                IO.Log.debug('Current mode changed into "CROSSVALIDATION"')
            elif value.lower() == 'prediction':
                self.__current_mode = 'prediction'
                IO.Log.debug('Current mode changed into "PREDICTION"')
            elif value.lower() == 'start':
                self.__current_mode = 'start'
                IO.Log.debug('Current mode changed into "START"')
            else:
                IO.Log.error('Unknown mode ({}) passed to '
                             'currentMode()'.format(value))
                return
            self.CurrentModeLabel.setText(self.__current_mode.capitalize()
                                          + ' mode')
        else:
            IO.Log.error('currentMode() takes a string when used as a setter')

    def newModel(self, load=False):
        """Both getter and setter of pls-da model."""
        if self.__plsda_model is not None:
            title = 'Replace current model?'
            msg = str('Are you sure to replace the current model? '
                      '(All data not saved will be lost)')
            IO.Log.debug(title)
            if not popupQuestion(msg, title=title, parent=self.MainWindow):
                IO.Log.debug('NO (not replacing current model)')
                return
            IO.Log.debug('YES (replacing current model)')

        if load:
            input_file = popupChooseFile(input=True,
                                         name_filter=self.pickle_file_filter,
                                         parent=self.MainWindow)
        else:
            input_file = popupChooseFile(input=True,
                                         name_filter=self.csv_file_filter,
                                         parent=self.MainWindow)
        if input_file is None:
            return

        try:
            if load:
                with open(input_file, 'br') as f:
                    self.__plsda_model = pickle.load(f)
            else:
                self.__plsda_model = model.PLS_DA(csv_file=input_file)
        except Exception as e:
            self.__plsda_model = None
            IO.Log.debug(str(e))
            popupError(message=str(e), parent=self.MainWindow)
            return
        if not load:
            title = 'Choose preprocessing'
            msg = 'Please choose the desired preprocessing: '
            choices = (('preprocess_autoscale', 'autoscaling'),
                       ('preprocess_mean', 'centering'),
                       ('preprocess_normalize', 'normalizing'),
                       ('no_preprocessing', 'none'))
            IO.Log.debug(title)
            ok, index = popupChooseItem(msg, [b for a, b in choices],
                                        title=title, parent=self.MainWindow)
            for i, (attr, prep) in enumerate(choices):
                if ok and index == i:
                    IO.Log.debug('OK (chosen preprocessing: {})'.format(prep))
                    getattr(self.__plsda_model, attr)(use_original=True)
                    break
            else:
                IO.Log.debug('CANCEL (not chosen any preprocessing)')
            self.__plsda_model.nipals_method()
            IO.Log.debug('Model created correctly')
        else:
            IO.Log.debug('Model loaded correctly')
        self.currentMode('model')

    def saveModel(self):
        if self.__plsda_model is None:
            msg = 'To save a model you have to create or load it before'
            IO.Log.debug(msg)
            popupError(msg, parent=self.MainWindow)
            return

        output_file = popupChooseFile(output=True,
                                      name_filter=self.pickle_file_filter,
                                      parent=self.MainWindow)
        if output_file is None:
            return
        if not output_file.endswith('.p') and \
           not output_file.endswith('.pkl') and \
           not output_file.endswith('.pickle'):
            dirname, basename = os.path.split(output_file)
            if '.' in basename:
                basename = '.'.join(basename.split('.')[:-1])
            basename += '.p'
            output_file = os.path.join(dirname, basename)
            IO.Log.debug('Adapted output file', output_file)
            if os.path.isfile(output_file):
                IO.Log.debug('Output file already exists, '
                             'it will be overwritten!')

        try:
            with open(output_file, 'bw') as f:
                pickle.dump(self.__plsda_model, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.__plsda_model = None
            IO.Log.debug(str(e))
            popupError(message=str(e), parent=self.MainWindow)
            return
        IO.Log.debug('Model saved correctly')

    def setupStatusAttributes(self):
        self.__plsda_model = None
        self.currentMode('start')

    def setupHandlers(self):
        self.ActionModel.triggered.connect(lambda: self.currentMode('model'))
        self.ActionCV.triggered.connect(lambda: self.currentMode('cv'))
        self.ActionPrediction.triggered.connect(lambda:
                                                self.currentMode('prediction'))

        self.ActionNewModel.triggered.connect(lambda:
                                              self.newModel(load=False))
        self.ActionSaveModel.triggered.connect(self.saveModel)
        self.ActionLoadModel.triggered.connect(lambda:
                                               self.newModel(load=True))

    def show(self):
        self.MainWindow.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = UserInterface()
    ui.show()
    if utility.CLI.args().verbose:
        ui.MainWindow.dumpObjectTree()
        #  ui.MainWindow.dumpObjectInfo()
    sys.exit(app.exec_())
