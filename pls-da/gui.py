#!/usr/bin/env python3
# coding: utf-8

import copy
import enum
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import sys

import IO
import model
import plot
import utility


def new_qt(widget, name, parent=None):
    """Return a widget of type specified with 1st argument (a string).

       Also set the object name and optionally the parent widjet.
    """
    ret = getattr(QtWidgets, widget)(parent)
    ret.setObjectName(name)
    return ret


def clear(layout):
    """Recursively call deleteLayer() over all widgets in layout."""
    if not isinstance(layout, QtWidgets.QLayout):
        return
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().deleteLater()
        elif child.layout() is not None:
            clear(child.layout())


def _popup_choose(parent, filter_csv=False,
                  _input=False, _output=False, _directory=False, _file=False):
    """Display a dialog to choose an input/output file/directory.

       Return file path or None.
    """
    if bool(_input) == bool(_output):
        IO.Log.error('In _popup_choose() input/output flags must differ')
        return
    if bool(_file) == bool(_directory):
        IO.Log.error('In _popup_choose() file/directory flags must differ')
        return

    mode = 'input' if _input else 'output'
    obj = 'file' if _file else 'directory'

    IO.Log.debug('Choose an {} {}'.format(mode, obj))
    dialog = new_qt('QFileDialog',
                 'popupChoose' + mode.capitalize() + obj.capitalize(),
                 parent=parent)
    dialog.setWindowTitle('Choose an {} {}'.format(mode, obj))

    if _input:
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
    else:  # _output
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)

    if _file:
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if filter_csv:
            dialog.setNameFilter("Comma-separated values files (*.csv *.txt)")
    else:  # _directory
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly)

    if not dialog.exec():
        IO.Log.debug('CANCEL (not chosen {} {})'.format(mode, obj))
        return None
    IO.Log.debug('OK (chosen {} {})'.format(mode, obj),
                 dialog.selectedFiles()[0])
    return dialog.selectedFiles()[0]


def popup_choose_input_directory(parent):
    """Display a dialog to choose an input directory; return its path or None.
    """
    return _popup_choose(parent, _input=True, _directory=True)


def popup_choose_output_directory(parent):
    """Display a dialog to choose an output directory; return its path or None.
    """
    return _popup_choose(parent, _output=True, _directory=True)


def popup_choose_input_file(parent, filter_csv=False):
    """Display a dialog to choose an input file; return its path or None."""
    return _popup_choose(parent, filter_csv, _input=True, _file=True)


def popup_choose_output_file(parent, filter_csv=False):
    """Display a dialog to choose an output file; return its path or None."""
    return _popup_choose(parent, filter_csv, _output=True, _file=True)


def popup_choose_item(message, item_list, parent, title=None):
    """Display a dialog to choose an item from a list.

       Return tuple with two values:
         True on Ok answer, False otherwise
         index of the chosen item in item_list.
    """
    dialog = new_qt('QInputDialog', 'popup_choose_item', parent=parent)
    if title is not None:
        dialog.setWindowTitle(title)
    dialog.setLabelText(message)
    dialog.setInputMode(QtWidgets.QInputDialog.TextInput)
    dialog.setComboBoxItems(item_list)
    ok_answer = dialog.exec() == QtWidgets.QDialog.Accepted
    return ok_answer, item_list.index(dialog.textValue())


def popup_error(message, parent):
    """Display a dialog with an informative message and an Ok button."""
    dialog = new_qt('QMessageBox', 'popup_error', parent=parent)
    dialog.setIcon(QtWidgets.QMessageBox.Critical)
    if not isinstance(message, str):
        message = str(message)
    dialog.setText(message)
    dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
    dialog.exec()


def popup_question(message, parent, title=None):
    """Display a dialog with a question.

       Return True on Yes answer, False otherwise.
    """
    dialog = new_qt('QMessageBox', 'popup_question', parent=parent)
    if title is not None:
        dialog.setWindowTitle(title)
    dialog.setText(message)
    dialog.setStandardButtons(QtWidgets.QMessageBox.Yes
                              | QtWidgets.QMessageBox.No)
    choice = dialog.exec()
    return choice == QtWidgets.QMessageBox.Yes


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


def set_policy(widget, h_policy='Preferred', v_policy='Preferred',
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


class Lane(enum.Enum):
    """Enumerate to identify the lanes of the gui which can contain plots."""

    Left = 'Left'
    Central = 'Central'


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
        set_policy(self.MainWindow, 'Expanding', 'Expanding', 0, 0)
        set_size(self.MainWindow, minimum=(800, 600), maximum=(7680, 4320))
        self.MainWindow.setUnifiedTitleAndToolBarOnMac(True)

        # Previously in setupUi()
        self.MainWidget = new_qt('QWidget', 'MainWidget', parent=self.MainWindow)
        set_policy(self.MainWidget, 'Preferred', 'Preferred', 0, 0)
        set_size(self.MainWidget, minimum=(800, 600), maximum=(7680, 4300))

        self.MainSplitter = new_qt('QSplitter', 'MainSplitter',
                                parent=self.MainWidget)
        set_policy(self.MainSplitter, 'Expanding', 'Expanding', 0, 0)
        set_size(self.MainSplitter, minimum=(800, 600), maximum=(7680, 4300))
        self.MainSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.MainSplitter.setHandleWidth(3)

        # Start creating widgets to put inside LeftWidget
        self.LeftScrollAreaWidgetContents = new_qt('QWidget',
                                                'LeftScrollAreaWidgetContents')
        self.LeftScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0,
                                                                   291, 565))
        set_policy(self.LeftScrollAreaWidgetContents,
                   'Expanding', 'Expanding', 0, 0)
        set_size(self.LeftScrollAreaWidgetContents,
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

        self.add_label(Lane.Left, row=0, name='LVs', text='Latent Variables')
        self.add_spin_box(Lane.Left, row=0, name='LVs')

        self.add_radio_button(Lane.Left, row=1, name='X',
                              group_name='LeftButtonGroup')
        self.add_spin_box(Lane.Left, row=1, name='X')

        self.add_radio_button(Lane.Left, row=2, name='Y',
                              group_name='LeftButtonGroup')
        self.add_spin_box(Lane.Left, row=2, name='Y')

        self.add_push_button(Lane.Left, row=3, name='Back',
                             role=QtWidgets.QFormLayout.LabelRole)
        self.add_push_button(Lane.Left, row=3, name='Plot')

        self.LeftWidget = new_qt('QWidget', 'LeftWidget',
                              parent=self.MainSplitter)
        set_size(self.LeftWidget, minimum=(200, 580), maximum=(3637, 4300))

        self.LeftComboBox = new_qt('QComboBox', 'LeftComboBox',
                                parent=self.LeftWidget)
        set_size(self.LeftComboBox, minimum=(194, 22), maximum=(3631, 22))
        for entry in self.drop_down_choices:
            self.LeftComboBox.addItem("")

        self.LeftScrollArea = new_qt('QScrollArea', 'LeftScrollArea',
                                  parent=self.LeftWidget)
        set_size(self.LeftScrollArea, minimum=(194, 547), maximum=(3631, 4267))
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
        set_policy(self.CentralScrollAreaWidgetContents,
                   'Expanding', 'Expanding', 0, 0)
        set_size(self.CentralScrollAreaWidgetContents,
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

        self.add_label(Lane.Central, row=0, name='LVs',
                       text='Latent Variables')
        self.add_spin_box(Lane.Central, row=0, name='LVs')

        self.add_spin_box(Lane.Central, row=1, name='X')
        self.add_radio_button(Lane.Central, row=1, name='X',
                              group_name='CentralButtonGroup')

        self.add_spin_box(Lane.Central, row=2, name='Y')
        self.add_radio_button(Lane.Central, row=2, name='Y',
                              group_name='CentralButtonGroup')

        self.add_push_button(Lane.Central, row=3, name='Back',
                             role=QtWidgets.QFormLayout.LabelRole)
        self.add_push_button(Lane.Central, row=3, name='Plot')

        self.CentralWidget = new_qt('QWidget', 'CentralWidget',
                                 parent=self.MainSplitter)
        set_size(self.CentralWidget, minimum=(200, 580), maximum=(3637, 4300))

        self.CentralComboBox = new_qt('QComboBox', 'CentralComboBox',
                                   parent=self.CentralWidget)
        set_policy(self.CentralComboBox, 'Expanding', 'Expanding', 0, 0)
        set_size(self.CentralComboBox, minimum=(194, 22), maximum=(3631, 22))
        for entry in self.drop_down_choices:
            self.CentralComboBox.addItem("")

        self.CentralScrollArea = new_qt('QScrollArea', 'CentralScrollArea',
                                     parent=self.CentralWidget)
        set_size(self.CentralScrollArea,
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
        set_policy(self.RightScrollAreaWidgetContents,
                   'Expanding', 'Expanding', 0, 0)
        set_size(self.RightScrollAreaWidgetContents,
                 minimum=(138, 534), maximum=(388, 4259))

        self.DetailsLabel = new_qt('QLabel', 'DetailsLabel',
                                parent=self.RightScrollAreaWidgetContents)
        set_policy(self.DetailsLabel, 'Expanding', 'Expanding', 0, 0)
        set_size(self.DetailsLabel, minimum=(138, 534), maximum=(388, 4259))
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
        set_policy(self.RightWidget, 'Expanding', 'Expanding', 0, 0)
        set_size(self.RightWidget, minimum=(150, 580), maximum=(400, 4300))

        self.RightScrollArea = new_qt('QScrollArea', 'RightScrollArea',
                                   parent=self.RightWidget)
        set_size(self.RightScrollArea,
                 minimum=(144, 547), maximum=(394, 4272))
        self.RightScrollArea.setWidgetResizable(True)
        self.RightScrollArea.setWidget(self.RightScrollAreaWidgetContents)

        self.CurrentModeLabel = new_qt('QLabel', 'CurrentModeLabel',
                                    parent=self.RightWidget)
        set_policy(self.CurrentModeLabel, 'Expanding', 'Expanding', 0, 0)
        set_size(self.CurrentModeLabel, minimum=(144, 22), maximum=(394, 22))
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
        set_size(self.TopMenuBar, minimum=(800, 20), maximum=(7680, 20))

        self.MenuOptions = new_qt('QMenu', 'MenuOptions', parent=self.TopMenuBar)
        set_size(self.MenuOptions, minimum=(100, 20), maximum=(960, 4300))

        self.MenuChangeMode = new_qt('QMenu', 'MenuChangeMode',
                                  parent=self.TopMenuBar)
        set_size(self.MenuChangeMode, minimum=(100, 20), maximum=(960, 4300))

        self.MenuAbout = new_qt('QMenu', 'MenuAbout', parent=self.TopMenuBar)
        set_size(self.MenuAbout, minimum=(100, 20), maximum=(960, 4300))

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

        self.connect_handlers()

        self.plsda_model = None
        self.current_mode = 'start'

    @property
    def current_mode(self):
        """Get value of CurrentModeLabel."""
        return self._current_mode.capitalize() + ' mode'

    @current_mode.setter
    def current_mode(self, value):
        """Set value of CurrentModeLabel and change."""
        if not isinstance(value, str) or \
            value.lower() not in ('start', 'cv', 'crossvalidation', 'model',
                                  'prediction'):
            IO.Log.error('Could not change current mode to '
                         '{} !'.format(repr(value)))
            return
        self._current_mode = value.lower()
        if value.lower() == 'cv':
            self._current_mode = 'crossvalidation'

        IO.Log.debug('Current mode changed to: ' + self._current_mode.upper())
        self.CurrentModeLabel.setText(self.current_mode)

        model_flag, cv_flag, pred_flag = False, False, False
        if self._current_mode == 'crossvalidation':
            model_flag, pred_flag = True, True
        elif self._current_mode == 'model':
            cv_flag, pred_flag = True, True
        elif self._current_mode == 'prediction':
            model_flag, cv_flag = True, True

        self.ActionModel.setEnabled(model_flag)
        self.ActionCV.setEnabled(cv_flag)
        self.ActionPrediction.setEnabled(pred_flag)

        if self._current_mode == 'start':
            self.ActionSaveModel.setEnabled(False)
            self.ActionExport.setEnabled(False)
        else:
            self.ActionSaveModel.setEnabled(True)
            self.ActionExport.setEnabled(True)


    def _replace_current_model(self):
        """Show a popup to ask if plsda_model should be replaced."""
        if self.plsda_model is None:
            return True

        title = 'Replace current model?'
        msg = str('Are you sure to replace the current model? '
                  '(All data not saved will be lost)')
        IO.Log.debug(title)
        if popup_question(msg, title=title, parent=self.MainWindow):
            IO.Log.debug('YES (replacing current model)')
            return True
        else:
            IO.Log.debug('NO (not replacing current model)')
            return False

    def add_label(self, lane, row, name, text, word_wrap=True,
                  text_format=QtCore.Qt.AutoText,
                  alignment=QtCore.Qt.AlignCenter,
                  role=QtWidgets.QFormLayout.LabelRole):
        """Add to [Left|Central]PlotFormLayout a QLabel in row/role position.
        """
        attr_name = lane.value + str(name) + 'Label'
        parent_name = getattr(self, lane.value + 'ScrollAreaWidgetContents')

        new_label = new_qt('QLabel', attr_name, parent=parent_name)
        new_label.setTextFormat(text_format)
        new_label.setAlignment(alignment)
        new_label.setWordWrap(word_wrap)
        new_label.setText(str(text))

        set_policy(new_label, 'Preferred', 'Preferred', 0, 0)
        set_size(new_label, minimum=(70, 22), maximum=(1310, 170))

        setattr(self, attr_name, new_label)

        layout = getattr(self, lane.value + 'PlotFormLayout')
        layout.setWidget(row, role, new_label)

    def add_push_button(self, lane, row, name, text=None,
                        role=QtWidgets.QFormLayout.FieldRole):
        """Add to [Left|Central]PlotFormLayout a QPushButton in row/role pos.
        """
        attr_name = lane.value + str(name) + 'PushButton'
        parent_name = getattr(self, lane.value + 'ScrollAreaWidgetContents')

        new_push_button = new_qt('QPushButton', attr_name, parent=parent_name)
        new_push_button.setText(str(text if text is not None else name))

        set_policy(new_push_button, 'Preferred', 'Preferred', 0, 0)
        set_size(new_push_button, minimum=(70, 22), maximum=(1310, 170))

        setattr(self, attr_name, new_push_button)

        layout = getattr(self, lane.value + 'PlotFormLayout')
        layout.setWidget(row, role, new_push_button)

    def add_radio_button(self, lane, row, name, group_name, text=None,
                         role=QtWidgets.QFormLayout.LabelRole):
        """Add to [Left|Central]PlotFormLayout a QRadioButton in row/role pos.

           The QButtonGroup is searched by group_name and if not found it is
           created and put in self.group_name
        """
        attr_name = lane.value + str(name) + 'RadioButton'
        parent_name = getattr(self, lane.value + 'ScrollAreaWidgetContents')

        new_radio_button = new_qt('QRadioButton', attr_name, parent=parent_name)
        new_radio_button.setText(str(text if text is not None else name))

        set_policy(new_radio_button, 'Preferred', 'Preferred', 0, 0)
        set_size(new_radio_button, minimum=(70, 22), maximum=(1310, 170))

        group = getattr(self, group_name, None)
        if group is None:
            group = new_qt('QButtonGroup', group_name, parent=parent_name)
            setattr(self, group_name, group)
        group.addButton(new_radio_button)

        setattr(self, attr_name, new_radio_button)

        layout = getattr(self, lane.value + 'PlotFormLayout')
        layout.setWidget(row, role, new_radio_button)

    def add_spin_box(self, lane, row, name, minimum=1, maximum=99,
                     role=QtWidgets.QFormLayout.FieldRole):
        """Add to [Left|Central]PlotFormLayout a QSpinBox in row/role position.
        """
        attr_name = lane.value + str(name) + 'SpinBox'
        parent_name = getattr(self, lane.value + 'ScrollAreaWidgetContents')

        new_spin_box = new_qt('QSpinBox', attr_name, parent=parent_name)
        new_spin_box.setMinimum(minimum)
        new_spin_box.setMaximum(maximum)

        set_policy(new_spin_box, 'Preferred', 'Preferred', 0, 0)
        set_size(new_spin_box, minimum=(70, 22), maximum=(1310, 170))

        setattr(self, attr_name, new_spin_box)

        layout = getattr(self, lane.value + 'PlotFormLayout')
        layout.setWidget(row, role, new_spin_box)

    def new_model(self):
        """Initialize plsda_model attribute from csv."""
        if not self._replace_current_model():
            return

        input_file = popup_choose_input_file(parent=self.MainWindow,
                                             filter_csv=True)
        if input_file is None:
            return

        try:
            preproc = model.Preprocessing(input_file)
        except Exception as e:
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
            return

        title = 'Choose preprocessing'
        msg = 'Please choose the desired preprocessing: '
        choices = (('autoscale', 'autoscaling'),
                   ('center', 'centering'),
                   ('normalize', 'normalizing'),
                   ('empty_method', 'none'))
        IO.Log.debug(title)
        ok, index = popup_choose_item(msg, [b for a, b in choices],
                                      title=title, parent=self.MainWindow)
        for i, (method, name) in enumerate(choices):
            if ok and index == i:
                IO.Log.debug('OK (chosen preprocessing: {})'.format(name))
                getattr(preproc, method)()
                break
        else:
            IO.Log.debug('CANCEL (not chosen any preprocessing)')

        try:
            plsda_model = model.nipals(preproc.dataset, preproc.dummy_y)
        except Exception as e:
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
            return

        self.preproc, self.plsda_model = preproc, plsda_model

        plot.update_global_preproc(self.preproc)
        plot.update_global_model(self.plsda_model)

        IO.Log.debug('Model created correctly')
        self.current_mode = 'model'

    def save_model(self):
        export_dir = popup_choose_output_directory(parent=self.MainWindow)
        if export_dir is None:
            return
        try:
            IO.dump(self.plsda_model, export_dir)
        except Exception as e:
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
            return
        IO.Log.debug('Model saved correctly')

    def load_model(self):
        """Initialize plsda_model attribute from workspace directory."""
        if not self._replace_current_model():
            return

        workspace_dir = popup_choose_input_directory(parent=self.MainWindow)
        if workspace_dir is None:
            return

        plsda_model_copy = copy.deepcopy(self.plsda_model)
        try:
            self.plsda_model = IO.load(workspace_dir)
        except Exception as e:
            self.plsda_model = plsda_model_copy
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
            return

        IO.Log.debug('Model loaded correctly')
        self.current_mode = 'model'

    def connect_handlers(self):
        self.ActionModel.triggered.connect(
                lambda: setattr(self, 'current_mode', 'model'))

        self.ActionCV.triggered.connect(
                lambda: setattr(self, 'current_mode', 'cv'))

        self.ActionPrediction.triggered.connect(
                lambda: setattr(self, 'current_mode', 'prediction'))

        self.ActionNewModel.triggered.connect(self.new_model)
        self.ActionSaveModel.triggered.connect(self.save_model)
        self.ActionLoadModel.triggered.connect(self.load_model)

        self.ActionExport.triggered.connect(
                lambda: popup_error('exception.NotImplementedError', parent=self.MainWindow))

#        self.LeftComboBox.currentIndexChanged.connect(
#                lambda index: print(repr(index)))
#        self.CentralComboBox.currentTextChanged.connect(
#                lambda text: print(repr(text)))

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
