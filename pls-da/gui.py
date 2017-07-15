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

    def __init__(self, main_window_title):
        self.setattr('MainWindow', 'QMainWindow',
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(800, 600), maximum=(7680, 4320)))
        self.MainWindow.setUnifiedTitleAndToolBarOnMac(True)
        self.MainWindow.setWindowTitle(main_window_title)
        self.MainWindow.setEnabled(True)
        self.MainWindow.resize(800, 600)

        self.setattr('MainWidget', 'QWidget', parent=self.MainWindow,
                     policy=('Preferred', 'Preferred', 0, 0),
                     size=dict(minimum=(800, 600), maximum=(7680, 4300)))

        self.setattr('MainGridLayout', 'QGridLayout', parent=self.MainWidget)
        self.MainGridLayout.setContentsMargins(0, 0, 0, 0)
        self.MainGridLayout.setSpacing(0)
        self.MainWindow.setCentralWidget(self.MainWidget)

        self.setattr('MainSplitter', 'QSplitter', parent=self.MainWidget,
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(800, 600), maximum=(7680, 4300)))
        self.MainGridLayout.addWidget(self.MainSplitter, 0, 0, 1, 1)
        self.MainSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.MainSplitter.setHandleWidth(3)

        # Lane.Left
        self.setattr('LeftWidget', 'QWidget', parent=self.MainSplitter,
                     size=dict(minimum=(200, 580), maximum=(3637, 4300)))

        self.setattr('LeftGridLayout', 'QGridLayout', parent=self.LeftWidget)
        self.LeftGridLayout.setContentsMargins(3, 3, 3, 3)
        self.LeftGridLayout.setSpacing(5)

        self.setattr('LeftComboBox', 'QComboBox', parent=self.LeftWidget,
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(194, 22), maximum=(3631, 22)))
        self.LeftGridLayout.addWidget(self.LeftComboBox, 0, 0, 1, 1)

        self.setattr('LeftScrollArea', 'QScrollArea', parent=self.LeftWidget,
                     size=dict(minimum=(194, 547), maximum=(3631, 4267)))
        self.LeftScrollArea.setWidgetResizable(True)
        self.LeftGridLayout.addWidget(self.LeftScrollArea, 1, 0, 1, 1)

        self.setattr('LeftScrollAreaWidgetContents', 'QWidget',
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(174, 427), maximum=(3611, 4147)))
        self.LeftScrollArea.setWidget(self.LeftScrollAreaWidgetContents)
        self.LeftScrollAreaWidgetContents.setGeometry(
                QtCore.QRect(0, 0, 290, 565))
        self.LeftScrollAreaWidgetContents.setLayoutDirection(
                QtCore.Qt.LeftToRight)

        self.setattr('LeftPlotFormLayout', 'QFormLayout',
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

        # Lane.Central
        self.setattr('CentralWidget', 'QWidget', parent=self.MainSplitter,
                     size=dict(minimum=(200, 580), maximum=(3637, 4300)))

        self.setattr('CentralGridLayout', 'QGridLayout',
                     parent=self.CentralWidget)
        self.CentralGridLayout.setContentsMargins(3, 3, 3, 3)
        self.CentralGridLayout.setSpacing(5)

        self.setattr('CentralComboBox', 'QComboBox', parent=self.CentralWidget,
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(194, 22), maximum=(3631, 22)))
        self.CentralGridLayout.addWidget(self.CentralComboBox, 0, 0, 1, 1)

        self.setattr('CentralScrollArea', 'QScrollArea',
                     parent=self.CentralWidget,
                     size=dict(minimum=(194, 547), maximum=(3631, 4267)))
        self.CentralScrollArea.setWidgetResizable(True)
        self.CentralGridLayout.addWidget(self.CentralScrollArea, 1, 0, 1, 1)

        self.setattr('CentralScrollAreaWidgetContents', 'QWidget',
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(174, 427), maximum=(3611, 4147)))
        self.CentralScrollArea.setWidget(self.CentralScrollAreaWidgetContents)
        self.CentralScrollAreaWidgetContents.setGeometry(
                QtCore.QRect(0, 0, 290, 565))
        self.CentralScrollAreaWidgetContents.setLayoutDirection(
                QtCore.Qt.LeftToRight)

        self.setattr('CentralPlotFormLayout', 'QFormLayout',
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

        # Lane on the right
        self.setattr('RightWidget', 'QWidget', parent=self.MainSplitter,
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(150, 580), maximum=(400, 4300)))

        self.setattr('RightGridLayout', 'QGridLayout', parent=self.RightWidget)
        self.RightGridLayout.setContentsMargins(3, 3, 3, 3)
        self.RightGridLayout.setSpacing(5)

        self.setattr('CurrentModeLabel', 'QLabel', parent=self.RightWidget,
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(144, 22), maximum=(394, 22)))
        self.CurrentModeLabel.setLineWidth(1)
        self.CurrentModeLabel.setTextFormat(QtCore.Qt.AutoText)
        self.CurrentModeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.CurrentModeLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.CurrentModeLabel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.RightGridLayout.addWidget(self.CurrentModeLabel, 0, 0, 1, 1)

        self.setattr('RightScrollArea', 'QScrollArea', parent=self.RightWidget,
                     size=dict(minimum=(144, 547), maximum=(394, 4272)))
        self.RightScrollArea.setWidgetResizable(True)
        self.RightGridLayout.addWidget(self.RightScrollArea, 1, 0, 1, 1)

        self.setattr('RightScrollAreaWidgetContents', 'QWidget',
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(138, 534), maximum=(388, 4259)))
        self.RightScrollArea.setWidget(self.RightScrollAreaWidgetContents)
        self.RightScrollAreaWidgetContents.setGeometry(
                QtCore.QRect(0, 0, 189, 565))

        self.setattr('RightDetailsGridLayout', 'QGridLayout',
                     parent=self.RightScrollAreaWidgetContents)
        self.RightDetailsGridLayout.setContentsMargins(3, 3, 3, 3)
        self.RightDetailsGridLayout.setSpacing(5)

        self.setattr('DetailsLabel', 'QLabel',
                     parent=self.RightScrollAreaWidgetContents,
                     policy=('Expanding', 'Expanding', 0, 0),
                     size=dict(minimum=(138, 534), maximum=(388, 4259)))
        self.RightDetailsGridLayout.addWidget(self.DetailsLabel, 0, 0, 1, 1)
        self.DetailsLabel.setAlignment(QtCore.Qt.AlignHCenter
                                       | QtCore.Qt.AlignTop)
        self.DetailsLabel.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByKeyboard
                | QtCore.Qt.TextSelectableByMouse)
        self.DetailsLabel.setText("Details")
        self.DetailsLabel.setWordWrap(True)

        # MenuBar
        self.setattr('TopMenuBar', 'QMenuBar', parent=self.MainWindow,
                     size=dict(minimum=(800, 20), maximum=(7680, 20)))
        self.TopMenuBar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.MainWindow.setMenuBar(self.TopMenuBar)

        for menu in self.menu_bar:
            self.setattr(menu['name'], 'QMenu', parent=self.TopMenuBar,
                         size=dict(minimum=(100, 20), maximum=(960, 4300)))
            menu_obj = getattr(self, menu['name'])
            self.TopMenuBar.addAction(menu_obj.menuAction())
            menu_obj.setTitle(menu['title'])

            for action in self.menu_action(menu['name']):
                print(action['name'])
                self.setattr(action['name'], 'QAction', parent=menu_obj)
                action_obj = getattr(self, action['name'])
                if action['shortcut'] is None:
                    action_obj.setSeparator(True)
                    action_obj.setText('')
                else:
                    action_obj.setText(action['text'])
                    action_obj.setShortcut(action['shortcut'])
                menu_obj.addAction(action_obj)

        # Previously in retranslateUi()
        for entry in self.drop_down_menu:
            self.LeftComboBox.addItem(entry['text'])
            self.CentralComboBox.addItem(entry['text'])

        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

        self.connect_handlers()

        self.plsda_model = None
        self.current_mode = 'start'

    @property
    def drop_down_menu(self):
        """Return a generator iterator over drop down menu item properties.

           https://docs.python.org/3/glossary.html#term-generator-iterator
        """
        tmp = (('Scree', 'plot_scree'),
               ('Cumulative explained variance', 'plot_explained_variance'),
               ('Inner relationships', 'plot_inner_relations'),
               ('Scores', 'plot_scores'),
               ('Loadings', 'plot_loadings'),
               ('Biplot', 'plot_biplot'),
               ('Scores & Loadings', 'plot_scores_and_loadings'),
               ('Calculated Y', 'plot_calculated_y'),
               ('Predicted Y', 'plot_predicted_y'),
               ('T² – Q', 'plot_t_square_q'),
               ('Residuals – Leverage', 'plot_residuals_leverage'),
               ('Regression coefficients', 'plot_regression_coefficients'))
        for index, (text, method) in enumerate(tmp):
            yield {'index': index, 'text': text, 'method': method}

    @property
    def menu_bar(self):
        """Return a generator iterator over menu items properties.

           https://docs.python.org/3/glossary.html#term-generator-iterator
        """
        for name in ('Options', 'Change Mode', 'About'):
            yield {'name': 'Menu' + name.replace(' ', ''), 'title': '&' + name}

    def menu_action(self, menu):
        menu = menu.lstrip('Menu').replace(' ', '').replace('&', '')
        if menu == 'Options':
            tmp = (('&New model', 'Ctrl+N'),
                   ('&Save model', 'Ctrl+S'),
                   ('&Load model', 'Ctrl+L'),
                   ('Separator1', None),
                   ('&Export matrices', 'Ctrl+E'),
                   ('Separator2', None),
                   ('&Quit', 'Ctrl+Q'))
        elif menu == 'ChangeMode':
            tmp = (('&Model', 'Ctrl+M'),
                   ('Cross&Validation', 'Ctrl+V'),
                   ('&Prediction', 'Ctrl+P'))
        elif menu == 'About':
            tmp = (('A&bout this project', 'F1'),
                   ('Abo&ut Qt', 'F2'))
        for text, shortcut in tmp:
            l = [word.capitalize() for word in text.replace('&', '').split()]
            yield {'name': 'Action' + ''.join(l).replace('sv', 'sV'),
                   'text': text, 'shortcut': shortcut}

    @property
    def current_mode(self):
        """Get value of CurrentModeLabel."""
        return self._current_mode.capitalize().replace('sv', 'sV') + ' mode'

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
        self.ActionCrossValidation.setEnabled(cv_flag)
        self.ActionPrediction.setEnabled(pred_flag)

        if self._current_mode == 'start':
            self.ActionSaveModel.setEnabled(False)
            self.ActionExportMatrices.setEnabled(False)
        else:
            self.ActionSaveModel.setEnabled(True)
            self.ActionExportMatrices.setEnabled(True)


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

    def setattr(self, name, widget, parent=None, policy=None, size=None):
        """Wrapper of qt5 module function and setattr over self."""
        new_widget = new_qt(widget=widget, name=name, parent=parent)
        setattr(self, name, new_widget)
        if policy is not None and isinstance(policy, tuple):
            set_policy(new_widget, *policy)
        if size is not None and isinstance(size, dict):
            set_size(new_widget, **size)

    def add_label(self, lane, row, name, text, word_wrap=True,
                  text_format=QtCore.Qt.AutoText,
                  alignment=QtCore.Qt.AlignLeft,
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
        return new_label

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
        return new_push_button

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
        return new_radio_button

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
        return new_spin_box

    def call_plot_method(self, lane, index=None, text=None):
        if index is None and text is None:
            IO.Log.warning('UserInterface.call_plot_method() got None index '
                           'and None text!')
            return
        for entry in self.drop_down_menu:
            if index == entry['index'] or text == entry['text']:
                # clear layouts from previous widget
                # clear(getattr(self, lane.value + 'PlotFormLayout'))
                # populate layouts with necessary widget
                return getattr(self, entry['method'])(lane)

    def plot_scree(self, lane):
        pass

    def plot_explained_variance(self, lane):
        pass

    def plot_inner_relations(self, lane):
        pass

    def plot_scores(self, lane):
        pass

    def plot_loadings(self, lane):
        pass

    def plot_biplot(self, lane):
        pass

    def plot_scores_and_loadings(self, lane):
        pass

    def plot_calculated_y(self, lane):
        pass

    def plot_predicted_y(self, lane):
        pass

    def plot_t_square_q(self, lane):
        pass

    def plot_residuals_leverage(self, lane):
        pass

    def plot_regression_coefficients(self, lane):
        pass

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
        self.ActionNewModel.triggered.connect(self.new_model)
        self.ActionSaveModel.triggered.connect(self.save_model)
        self.ActionLoadModel.triggered.connect(self.load_model)

        self.ActionExportMatrices.triggered.connect(
                lambda: popup_error('exception.NotImplementedError', parent=self.MainWindow))

        self.ActionQuit.triggered.connect(self.quit)

        self.ActionModel.triggered.connect(
                lambda: setattr(self, 'current_mode', 'model'))
        self.ActionCrossValidation.triggered.connect(
                lambda: setattr(self, 'current_mode', 'cv'))
        self.ActionPrediction.triggered.connect(
                lambda: setattr(self, 'current_mode', 'prediction'))

        self.ActionAboutQt.triggered.connect(QtWidgets.QApplication.aboutQt)

        self.LeftComboBox.currentIndexChanged.connect(
                lambda idx: self.call_plot_method(Lane.Left, index=idx))
        self.CentralComboBox.currentTextChanged.connect(
                lambda txt: self.call_plot_method(Lane.Central, text=txt))

    def quit(self):
        """Ask for confirmation with a popup and quit returning 0."""
        if popup_question('Would you really like to quit?', title='Quit',
                          parent=self.MainWindow):
            QtCore.QCoreApplication.quit()

    def show(self):
        self.MainWindow.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = UserInterface('PLS-DA')
    ui.show()
    if utility.CLI.args().verbose:
        ui.MainWindow.dumpObjectTree()
        #  ui.MainWindow.dumpObjectInfo()
    sys.exit(app.exec_())
