#!/usr/bin/env python3
# coding: utf-8

""" PLS-DA is a project about the Partial least squares Discriminant Analysis
    on a given dataset.'
    PLS-DA is a project developed for the Processing of Scientific Data exam
    at University of Modena and Reggio Emilia.
    Copyright (C) 2017  Serena Ziviani, Federico Motta
    This file is part of PLS-DA.
    PLS-DA is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.
    PLS-DA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with PLS-DA.  If not, see <http://www.gnu.org/licenses/>.
"""

__authors__ = "Serena Ziviani, Federico Motta"
__copyright__ = "PLS-DA  Copyright (C)  2017"
__license__ = "GPL3"


import copy
import enum
import traceback
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as Toolbar
from PyQt5.QtCore import QCoreApplication, QMetaObject, QRect, QSize, Qt
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import (QAction, QApplication, QButtonGroup, QCheckBox,
                             QComboBox, QDialog, QFileDialog, QFormLayout,
                             QGridLayout, QFrame, QInputDialog, QLabel,
                             QLayout, QMainWindow, QMenu, QMenuBar,
                             QMessageBox, QPushButton, QRadioButton,
                             QScrollArea, QSizePolicy as Policy, QSpinBox,
                             QSplitter, QVBoxLayout, QWidget)

import IO
import model
import plot
import utility


if __name__ == '__main__':
    raise SystemExit('Please do not run that script, load it!')


def clear(layout):
    """Recursively call delete() over all widgets in layout."""
    if not isinstance(layout, QLayout):
        return
    try:
        layout_name = layout.objectName()
    except RuntimeError as e:
        if 'wrapped C/C++ object of type' in str(e) and \
           'has been deleted' in str(e):
            # layout has already been garbage collected
            return
        else:
            raise e
    else:
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                delete(child.widget())
            elif child.layout() is not None:
                clear(child.layout())
        if layout_name.strip():
            IO.Log.debug('Cleared {}'.format(layout_name))


def delete(widget):
    """Wrap deleteLater() over the widget and log event on debug."""
    if not isinstance(widget, QWidget):
        return
    try:
        widget_name = widget.objectName()
    except RuntimeError as e:
        if 'wrapped C/C++ object of type' in str(e) and \
           'has been deleted' in str(e):
            # widget has already been garbage collected
            return
        else:
            raise e
    else:
        widget.deleteLater()
        if widget_name.strip():
            IO.Log.debug('Deleted {}'.format(widget_name))


def change_enable_flag(item, enabled):
    """Change enabled state of QStandardItem."""
    if not isinstance(item, QStandardItem):
        raise TypeError('bad item type in change_enabled_flag() '
                        '({})'.format(type(item)))
    if enabled:
        item.setFlags(item.flags() | Qt.ItemIsEnabled)
    else:
        # disabled
        item.setFlags(item.flags() & ~Qt.ItemIsEnabled)


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
    dialog = QFileDialog(parent)
    dialog.setObjectName('popupChoose' + mode.capitalize() + obj.capitalize())
    dialog.setWindowTitle('Choose an {} {}'.format(mode, obj))

    if _input:
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
    else:  # _output
        dialog.setAcceptMode(QFileDialog.AcceptSave)

    if _file:
        if _input:
            dialog.setFileMode(QFileDialog.ExistingFile)
        if filter_csv:
            dialog.setNameFilter('Comma-separated values files (*.csv *.txt)')
            if _output:
                dialog.setDefaultSuffix('csv')
    else:  # _directory
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly)

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
    dialog = QInputDialog(parent)
    dialog.setObjectName('popup_choose_item')
    dialog.setWindowTitle(title if title is not None else '')
    dialog.setLabelText(message)
    dialog.setInputMode(QInputDialog.TextInput)
    dialog.setComboBoxItems(item_list)
    ok_answer = dialog.exec() == QDialog.Accepted
    return ok_answer, item_list.index(dialog.textValue())


def popup_error(message, parent):
    """Display a dialog with an informative message and an Ok button."""
    dialog = QMessageBox(parent)
    dialog.setObjectName('popup_error')
    dialog.setIcon(QMessageBox.Critical)
    dialog.setText(str(message))
    dialog.setStandardButtons(QMessageBox.Ok)
    dialog.exec()


def popup_question(message, parent, title=None):
    """Display a dialog with a question.

       Return True on Yes answer, False otherwise.
    """
    dialog = QMessageBox(parent)
    dialog.setObjectName('popup_question')
    dialog.setWindowTitle(title if title is not None else '')
    dialog.setText(message)
    dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    return dialog.exec() == QMessageBox.Yes


class Column(enum.Enum):
    """Enumerate to identify the columns of the QFormLayout."""

    Left = QFormLayout.LabelRole
    Right = QFormLayout.FieldRole

    # WARNING: the spanning role breaks the central alignment of QFormLayout !
    Both = QFormLayout.SpanningRole


class Lane(enum.Enum):
    """Enumerate to identify the lanes of the gui which can contain plots."""

    Left = 'Left'
    Central = 'Central'
    Right = 'Right'

    def __str__(self):
        """The str() builtin will return enumerate value."""
        return self.value


class Mode(enum.Enum):
    """Enumerate to identify the 4 finite-states of the application."""

    Start = 'start'
    Model = 'model'
    Crossvalidation = 'cross-validation'
    CV = 'cross-validation'  # useful alias of Mode.Crossvalidation
    Prediction = 'prediction'

    def __eq__(self, other):
        """Comparisons done with == and != operators will be case insensitive.
        """
        return self.value.lower() == other.value.lower()

    def __str__(self):
        """The returned string is ready to be the CurrentModeLabel text."""
        return self.value.capitalize() + ' mode'


class Widget(enum.Enum):
    """Enumerate to identify the kind of QWigdet to put in a QScrollArea."""

    Form = 'Form'
    VBox = 'VBox'

    def __str__(self):
        """The str() builtin will return enumerate value."""
        return self.value


class UserInterface(object):

    def __init__(self, main_window_title):
        main_window = self.set_attr('', QMainWindow, parent=None,
                                    size=(800, 600, 7680, 4320))
        main_window.setUnifiedTitleAndToolBarOnMac(True)
        main_window.setWindowTitle(main_window_title)
        main_window.setEnabled(True)
        main_window.resize(800, 600)

        main_widget = self.set_attr('Main', QWidget, parent=main_window,
                                    policy=Policy.Preferred,
                                    size=(800, 600, 7680, 4300))

        main_layout = self.set_attr('Main', QGridLayout, parent=main_widget,
                                    policy=None)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_window.setCentralWidget(main_widget)

        main_splitter = self.set_attr('Main', QSplitter, parent=main_widget,
                                      size=(800, 600, 7680, 4300))
        main_layout.addWidget(main_splitter, 0, 0, 1, 1)
        main_splitter.setOrientation(Qt.Horizontal)
        main_splitter.setHandleWidth(3)

        for lane in (Lane.Left, Lane.Central):
            parent = self.set_attr(lane, QWidget, parent=main_splitter,
                                   policy=None, size=(200, 580, 3637, 4300))

            layout = self.set_attr(lane, QGridLayout, parent=parent,
                                   policy=None)
            layout.setContentsMargins(3, 3, 3, 3)
            layout.setSpacing(5)

            drop_down = self.set_attr(lane, QComboBox, parent,
                                      size=(194, 22, 3631, 22))
            layout.addWidget(drop_down, 0, 0, 1, 1)
            for entry in self.drop_down_menu:
                drop_down.addItem(entry['text'])
            drop_down.setCurrentIndex(-1)

            scroll_area = self.set_attr(lane, QScrollArea, parent, policy=None,
                                        size=(194, 547, 3631, 4267))
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area, 1, 0, 1, 1)

        lane = Lane.Right
        parent = self.set_attr(lane, QWidget, parent=main_splitter,
                               size=(150, 580, 400, 4300))

        layout = self.set_attr(lane, QGridLayout, parent=parent, policy=None)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(5)

        current_mode_label = self.set_attr(
            'CurrentMode', QLabel, parent=parent, size=(144, 22, 394, 22))
        current_mode_label.setLineWidth(1)
        current_mode_label.setTextFormat(Qt.AutoText)
        current_mode_label.setAlignment(Qt.AlignCenter)
        current_mode_label.setFrameShadow(QFrame.Plain)
        current_mode_label.setFrameShape(QFrame.StyledPanel)
        layout.addWidget(current_mode_label, 0, 0, 1, 1)

        right_scroll_area = self.set_attr(
            lane, QScrollArea, parent=parent,
            policy=None, size=(144, 547, 394, 4272))
        layout.addWidget(right_scroll_area, 1, 0, 1, 1)

        right_vbox_widget = self.set_attr(
            str(lane) + str(Widget.VBox), QWidget, parent=right_scroll_area,
            size=(138, 534, 388, 4259))
        right_vbox_widget.setGeometry(QRect(0, 0, 138, 534))
        right_vbox_widget.setLayoutDirection(Qt.LayoutDirectionAuto)
        right_scroll_area.setWidget(right_vbox_widget)
        right_scroll_area.setWidgetResizable(True)

        right_vbox_layout = self.set_attr(
            str(lane), QVBoxLayout, parent=right_vbox_widget, policy=None)
        right_vbox_layout.setSizeConstraint(QLayout.SetMaximumSize)
        right_vbox_layout.setContentsMargins(4, 4, 4, 4)
        right_vbox_layout.setSpacing(4)

        # Build three stacked areas in the right lane
        self.populate_right_vbox_widgets_and_layouts()

        # Fill them with their widgets
        self.populate_right_model_widget()
        self.populate_right_cv_widget()
        self.populate_right_prediction_widget()

        top_menu_bar = self.set_attr('Top', QMenuBar, parent=main_window,
                                     policy=None, size=(800, 20, 7680, 20))
        top_menu_bar.setGeometry(QRect(0, 0, 800, 20))
        main_window.setMenuBar(top_menu_bar)

        for menu in self.menu_bar:
            menu_obj = self.set_attr(menu['name'], QMenu, parent=top_menu_bar,
                                     policy=None, size=(100, 20, 960, 4300))
            top_menu_bar.addAction(menu_obj.menuAction())
            menu_obj.setTitle(menu['title'])

            for action in self.menu_action(menu['name']):
                action_obj = self.set_attr(action['name'], QAction,
                                           parent=menu_obj, policy=None)
                if action['shortcut'] is None:
                    action_obj.setSeparator(True)
                    action_obj.setText('')
                else:
                    action_obj.setShortcut(action['shortcut'])
                    action_obj.setText(action['text'])
                menu_obj.addAction(action_obj)

        QMetaObject.connectSlotsByName(main_window)
        self.connect_handlers()

        self.current_mode = Mode.Start

    @property
    def plsda_model(self):
        """Return reference to object of model.Model type or None."""
        return getattr(self, '_plsda_model', None)

    @plsda_model.setter
    def plsda_model(self, value):
        """Wrap update of plot.MODEL reference and refresh info in right area.

           Raises TypeError on bad value type.
        """
        # create attribute if not yet existent
        self._plsda_model = getattr(self, '_plsda_model', None)

        if not isinstance(value, model.Model) and value is not None:
            raise TypeError('value assigned to self.plsda_model is not an '
                            'instance of model.Model ({})'.format(type(value)))

        self._plsda_model = value
        plot.update_global_model(value)

        self.update_right_model_info()

    @property
    def train_set(self):
        """Return reference to object of model.TrainingSet type or None."""
        return getattr(self, '_train_set', None)

    @train_set.setter
    def train_set(self, value):
        """Wrap update of plot.TRAIN_SET reference.

           Raises TypeError on bad value type.
        """
        # create attribute if not yet existent
        self._train_set = getattr(self, '_train_set', None)

        if not isinstance(value, model.TrainingSet) and value is not None:
            raise TypeError('value assigned to self.train_set is not an '
                            'instance of model.TrainingSet '
                            '({})'.format(type(value)))

        self._train_set = value
        plot.update_global_train_set(value)

    @property
    def test_set(self):
        """Return reference to object of model.TestSet type or None."""
        return getattr(self, '_test_set', None)

    @test_set.setter
    def test_set(self, value):
        """Wrap update of plot.TEST_SET reference (and right prediction info).

           Raises TypeError on bad value type.
        """
        # create attribute if not yet existent
        self._test_set = getattr(self, '_test_set', None)

        if not isinstance(value, model.TestSet) and value is not None:
            raise TypeError('value assigned to self.test_set is not an '
                            'instance of model.TestSet '
                            '({})'.format(type(value)))

        self._test_set = value
        plot.update_global_test_set(value)

        self.update_right_prediction_info()

    @property
    def prediction_stats(self):
        """Return reference to object of model.Statistics type or None."""
        return getattr(self, '_stats', None)

    @prediction_stats.setter
    def prediction_stats(self, value):
        """Wrap update of plot.STATS reference.

           Raises TypeError on bad value type.
        """
        # create attribute if not yet existent
        self._stats = getattr(self, '_stats', None)

        if not isinstance(value, model.Statistics) and value is not None:
            raise TypeError('value assigned to self.stats is not an '
                            'instance of model.Statistics '
                            '({})'.format(type(value)))

        self._stats = value
        plot.update_global_statistics(value)
        self.change_plot_enabled_flag('RMSEP', True)
        self.change_plot_enabled_flag('Predicted Y – Real Y', True)
        self.change_plot_enabled_flag('Predicted Y', True)

    @property
    def cv_stats(self):
        """Return reference to a list of model.Statistics objects.

           Or None if the internal attribute is not set.
        """
        return getattr(self, '_cv_stats', None)

    @cv_stats.setter
    def cv_stats(self, value):
        """Ensure that value is a list of list."""
        if not isinstance(value, list) or not isinstance(value[0], list):
            raise TypeError('value assigned to self.cv_stats is not a list '
                            'of lists ({})'.format(repr(value)))
        for sublist in value:
            if not isinstance(sublist, list) or len(sublist) != len(value[0]):
                raise TypeError('value assigned to self.cv_stats is not '
                                'composed by lists of the same length '
                                '({})'.format(repr(value)))
        self._cv_stats = value
        self.change_plot_enabled_flag('RMSECV', True)
        self.update_right_cv_info()

    @property
    def drop_down_menu(self):
        """Return a generator iterator over drop down menu item properties."""
        tmp = (('Scree', 'scree'),
               ('Cumulative explained variance', 'explained_variance'),
               ('Inner relationships', 'inner_relations'),
               ('Scores', 'scores'),
               ('Loadings', 'loadings'),
               ('Biplot', 'biplot'),
               ('Scores & Loadings', 'scores_and_loadings'),
               ('Calculated Y', 'calculated_y'),
               ('Real Y – Predicted Y', 'predicted_y_real_y'),
               ('Predicted Y', 'predicted_y'),
               ('Samples – X residuals', 'x_residuals_over_samples'),
               ('Samples – Y residuals', 'y_residuals_over_samples'),
               ('Samples – Leverage', 'samples_leverage'),
               ('Q – Leverage', 'q_over_leverage'),
               ('Leverage – Y residuals', 'residuals_leverage'),
               ('T² – Q', 't_square_q'),
               ('Regression coefficients', 'regression_coefficients'),
               ('Weights Scatter Plot', 'weights'),
               ('Weights Line Plot', 'weights_line'),
               ('Data', 'data'),
               ('RMSEC', 'rmesec'),
               ('RMSECV', 'rmesecv'),
               ('RMSEP', 'rmesep'),
               )
        for index, (text, method) in enumerate(tmp):
            yield {'index': index, 'text': text, 'method': method + '_plot'}

    @property
    def menu_bar(self):
        """Return a generator iterator over menu items properties."""
        for name in ('File', 'Change Mode', 'About'):
            yield {'name': name.replace(' ', ''), 'title': '&' + name}

    def menu_action(self, menu):
        """Return a generator iterator over action items properties."""
        menu = menu.lstrip('Menu').replace(' ', '').replace('&', '')
        if menu == 'File':
            tmp = (('&New model', 'Ctrl+N'),
                   ('&Save model', 'Ctrl+S'),
                   ('&Load model', 'Ctrl+L'),
                   ('1_ Separator', None),
                   ('Load csv to &predict', 'Ctrl+R'),
                   ('2_ Separator', None),
                   ('&Export matrices', 'Ctrl+E'),
                   ('3_ Separator', None),
                   ('&Quit', 'Ctrl+Q'))
        elif menu == 'ChangeMode':
            tmp = (('&Model', 'Ctrl+M'),
                   ('Cross-&validation', 'Ctrl+V'),
                   ('&Prediction', 'Ctrl+P'))
        elif menu == 'About':
            tmp = (('A&bout this project', 'F1'),
                   ('Abo&ut Qt', 'F2'))
        for text, shortcut in tmp:
            l = [word.capitalize() for word in text.replace('&', '').split()]
            yield {'name': ''.join(l).replace('s-v', 'sV'),
                   'text': text, 'shortcut': shortcut}

    def back_button(self, lane):
        return getattr(self, str(lane) + 'BackPushButton')

    def canvas(self, lane):
        return getattr(self, str(lane) + 'Canvas')

    def figure(self, lane, tight_layout=True):
        ret = getattr(self, str(lane) + 'Figure', None)
        if ret is None:
            ret = plt.figure(tight_layout=tight_layout)
            setattr(self, str(lane) + 'Figure', ret)
        return ret

    def form_layout(self, lane, kind=None):
        if lane == Lane.Right and not isinstance(kind, Mode):
            raise TypeError(str('Please set "kind" argument in form_layout() '
                                'when lane is Lane.Right'))
        elif lane == Lane.Right:
            name = kind.value.capitalize() if kind != Mode.CV else 'CV'
            return getattr(self, str(lane) + name + 'FormLayout')
        else:
            return getattr(self, str(lane) + 'FormLayout')

    def form_widget(self, lane, kind=None):
        if lane == Lane.Right and not isinstance(kind, Mode):
            raise TypeError(str('Please set "kind" argument in form_widget() '
                                'when lane is Lane.Right'))
        elif lane == Lane.Right:
            name = kind.value.capitalize() if kind != Mode.CV else 'CV'
            return getattr(self, str(lane) + name + 'FormWidget')
        else:
            return getattr(self, str(lane) + 'FormWidget')

    def lva_spin_box(self, lane):
        return getattr(self, str(lane) + 'LVaSpinBox')

    def lvb_spin_box(self, lane):
        return getattr(self, str(lane) + 'LVbSpinBox')

    def lvs_spin_box(self, lane):
        return getattr(self, str(lane) + 'LVsSpinBox')

    def normalize_check_box(self, lane):
        return getattr(self, str(lane) + 'NormalizeCheckBox')

    def plot_button(self, lane):
        return getattr(self, str(lane) + 'PlotPushButton')

    def right_layout(self, kind):
        if not isinstance(kind, Mode):
            raise TypeError('kind parameter in right_layout() is not '
                            'instance of Mode ({})'.format(type(kind)))
        name = kind.value.Capitalize() if kind != Mode.CV else 'CV'
        return getattr(self, str(Lane.Right) + name + 'FormLayout')

    def right_widget(self, kind):
        if not isinstance(kind, Mode):
            raise TypeError('kind parameter in right_widget() is not '
                            'instance of Mode ({})'.format(type(kind)))
        name = kind.value.capitalize() if kind != Mode.CV else 'CV'
        return getattr(self, str(Lane.Right) + name + 'FormWidget')

    def right_model_lvs(self):
        return getattr(self, 'RightLVsModelSpinBox')

    def right_model_change_lvs_button(self):
        return getattr(self, 'RightLVsModelPushButton')

    def right_cv_start_button(self):
        return getattr(self, 'RightStartCVPushButton')

    def right_cv_samples(self):
        """Number of Samples in the SpinBox in the right CV area."""
        return getattr(self, 'RightSamplesCVSpinBox').value()

    def right_cv_splits(self):
        """Number of Splits in the SpinBox in the right CV area."""
        return getattr(self, 'RightSplitsCVSpinBox').value()

    def scroll_area(self, lane):
        return getattr(self, str(lane) + 'ScrollArea')

    def vbox_layout(self, lane):
        return getattr(self, str(lane) + 'VBoxLayout')

    def vbox_widget(self, lane):
        return getattr(self, str(lane) + 'VBoxWidget')

    def x_radio_button(self, lane):
        return getattr(self, str(lane) + 'XRadioButton')

    def y_radio_button(self, lane):
        return getattr(self, str(lane) + 'YRadioButton')

    @property
    def current_mode(self):
        """Return reference to gui.Mode object."""
        return self._current_mode

    @current_mode.setter
    def current_mode(self, value):
        """Change internal state of current_mode and text of CurrentModeLabel.

           Raises TypeError on bad value type.
        """
        if not isinstance(value, Mode):
            raise TypeError('value assigned to self.current_mode is not '
                            'an instance of gui.Mode ({})'.format(type(value)))

        # apply change in internal state and in text of the label
        self._current_mode = value
        self.CurrentModeLabel.setText(str(self.current_mode))
        IO.Log.debug('Current mode changed to: ' + str(self.current_mode))

        self.ModelAction.setEnabled(
            self.current_mode not in (Mode.Start, Mode.Model))
        self.CrossValidationAction.setEnabled(
            self.current_mode not in (Mode.Start, Mode.CV))
        self.PredictionAction.setEnabled(
            self.current_mode not in (Mode.Start, Mode.Prediction))

        if self.current_mode == Mode.Start:
            self.clear_plot_lanes_and_show_hints()
            self.change_plot_enabled_flag('RMSECV', False)
            self.change_plot_enabled_flag('RMSEP', False)
            self.change_plot_enabled_flag('Predicted Y – Real Y', False)
            self.change_plot_enabled_flag('Predicted Y', False)

        self.SaveModelAction.setEnabled(self.current_mode != Mode.Start)
        self.LoadCsvToPredictAction.setEnabled(self.current_mode != Mode.Start)
        self.ExportMatricesAction.setEnabled(self.current_mode != Mode.Start)
        self.LeftComboBox.setEnabled(self.current_mode != Mode.Start)
        self.CentralComboBox.setEnabled(self.current_mode != Mode.Start)

        self.update_right_model_lvs_spinbox(
            minimum=1, maximum=getattr(self.plsda_model, 'max_lv', 7),
            enabled=self.current_mode in (Mode.Model, Mode.CV))
        self.right_model_change_lvs_button().setEnabled(
            self.current_mode in (Mode.Model, Mode.CV))

        self.update_right_cv_splits_spinbox(
            minimum=2, maximum=getattr(self.plsda_model, 'n', 219),
            enabled=self.current_mode == Mode.CV)
        self.update_right_cv_samples_spinbox(
            minimum=2, maximum=getattr(self.plsda_model, 'n', 219),
            enabled=self.current_mode == Mode.CV)
        self.right_cv_start_button().setEnabled(self.current_mode == Mode.CV)

        if self.current_mode == Mode.Prediction and \
           (self.prediction_stats is None or self.test_set is None):
            if popup_question(message='Would you like to load a csv file to '
                                      'predict now?',
                              parent=self.MainWindow, title='Load test set'):
                self.load_csv_to_predict()

    def change_plot_enabled_flag(self, plot_name, enabled):
        """Enable or disable plot entry in drop down menus.

           Matching is done with the plot_name string.
        """
        for combobox in (self.LeftComboBox, self.CentralComboBox):
            for index, _ in enumerate(self.drop_down_menu):
                if combobox.itemText(index) == plot_name:
                    change_enable_flag(combobox.model().item(index), enabled)
                    break
            else:
                IO.Log.warning('Could not {}able plot named "{}"'.format(
                    'en' if enabled else 'dis', plot_name))

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
        IO.Log.debug('NO (not replacing current model)')
        return False

    def populate_right_vbox_widgets_and_layouts(self):
        lane = Lane.Right

        # DetailsLabel
        dl = self.set_attr('Details', QLabel, parent=self.vbox_widget(lane),
                           size=(108, 20, 358, 20))
        dl.setTextInteractionFlags(Qt.TextSelectableByKeyboard
                                   | Qt.TextSelectableByMouse)
        dl.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        dl.setText('Details')
        dl.setWordWrap(True)
        self.vbox_layout(lane).addWidget(dl)

        # Right [ Model | CV | Prediction ] Form [ Widget | Layout ]
        for index, kind in enumerate((Mode.Model, Mode.CV, Mode.Prediction)):
            separator = self.set_attr(str(lane) + 'Line' + str(index), QFrame,
                                      parent=self.vbox_widget(lane), size=None)
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            self.vbox_layout(lane).addWidget(separator)

            name = kind.value.capitalize() if kind != Mode.CV else 'CV'

            w = self.set_attr(str(lane) + name + str(Widget.Form), QWidget,
                              parent=self.vbox_widget(lane),
                              size=(108, 140, 358, 1406))
            l = self.set_attr(str(lane) + name, QFormLayout,
                              parent=self.right_widget(kind), policy=None)

            w.setGeometry(QRect(0, 0, 108, 140))
            w.setLayoutDirection(Qt.LeftToRight)
            l.setRowWrapPolicy(QFormLayout.DontWrapRows)
            l.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
            l.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
            l.setLabelAlignment(Qt.AlignLeft)
            l.setSizeConstraint(QLayout.SetMaximumSize)
            l.setSpacing(1)
            self.vbox_layout(lane).addWidget(w)

    def populate_right_model_widget(self, model_info=''):
        lane, parent = Lane.Right, self.right_widget(Mode.Model)

        self.add(QLabel, lane, Column.Both, row=0, name='Model', text='Model:',
                 label_alignment=Qt.AlignLeft, parent_widget=parent)

        self.add(QLabel, lane, Column.Left, row=1, name='LVs', text='LVs:',
                 word_wrap=False, label_alignment=Qt.AlignLeft,
                 parent_widget=parent)
        sb = self.add(QSpinBox, lane, Column.Right, row=1, name='LVs',
                      parent_widget=parent, size=(40, 25, 170, 520))
        sb.setEnabled(False)

        cb = self.add(QPushButton, lane, Column.Both, row=2, name='LVs',
                      text='Change LVs', parent_widget=parent,
                      size=(70, 25, 112, 520))
        cb.setEnabled(False)

        self.add(QLabel, lane, Column.Both, row=3, name='ModelInfo',
                 text=model_info, label_alignment=Qt.AlignLeft,
                 parent_widget=parent, size=(100, 30, 360, 1000))

    def populate_right_cv_widget(self, cv_info=''):
        lane, parent = Lane.Right, self.right_widget(Mode.CV)

        self.add(QLabel, lane, Column.Both, row=0, name='CV',
                 text='Cross-validation:', label_alignment=Qt.AlignLeft,
                 parent_widget=parent)

        self.add(QLabel, lane, Column.Left, row=1, name='Splits',
                 text='Splits:', word_wrap=False,
                 label_alignment=Qt.AlignLeft, parent_widget=parent)
        sb = self.add(QSpinBox, lane, Column.Right, row=1, name='Splits',
                      parent_widget=parent, size=(45, 25, 170, 520))
        sb.setEnabled(False)

        self.add(QLabel, lane, Column.Left, row=2, name='Samples',
                 text='Samples per blind:', word_wrap=True,
                 label_alignment=Qt.AlignLeft, parent_widget=parent,
                 size=(50, 40, 170, 520))
        sb = self.add(QSpinBox, lane, Column.Right, row=2, name='Samples',
                      parent_widget=parent, size=(45, 25, 170, 520))
        sb.setEnabled(False)

        self.add(QPushButton, lane, Column.Both, row=3, name='Start',
                 text='Start CV', parent_widget=parent,
                 size=(70, 25, 127, 520))

        self.add(QLabel, lane, Column.Both, row=4, name='CVInfo',
                 text=cv_info, label_alignment=Qt.AlignLeft,
                 parent_widget=parent, size=(100, 30, 360, 1000))

    def populate_right_prediction_widget(self, prediction_info=''):
        lane, parent = Lane.Right, self.right_widget(Mode.Prediction)

        self.add(QLabel, lane, Column.Both, row=0, name='Prediction',
                 text='Prediction:', label_alignment=Qt.AlignLeft,
                 parent_widget=parent)

        self.add(QLabel, lane, Column.Both, row=1, name='PredictionInfo',
                 text=prediction_info, label_alignment=Qt.AlignLeft,
                 parent_widget=parent, size=(100, 30, 360, 1000))

    def update_right_model_info(self):
        """Method to refresh the label with model infos."""
        l = getattr(self, 'RightModelInfoLabel', None)
        if l is not None:
            text = 'X-Block: {} x {}\n'.format(self.plsda_model.n,
                                               self.plsda_model.m)
            text += 'Y-Block: {} x {}\n'.format(self.plsda_model.n,
                                                self.plsda_model.p)
            s = model.Statistics(y_real=self.plsda_model.Y,
                                 y_pred=self.plsda_model.Y_modeled)
            text += 'RMSEC:\n{}\n'.format(utility.list_to_string(s.rmsec))
            text += 'R²:\n{}\n'.format(utility.list_to_string(s.r_squared))
            l.setText(text)

    def update_right_model_lvs_spinbox(self, minimum=None, maximum=None,
                                       enabled=False):
        """Change min, max values and enabled status."""
        sb = getattr(self, 'RightLVsModelSpinBox', None)
        if sb is not None:
            minimum = minimum if minimum is not None else sb.minimum
            maximum = maximum if maximum is not None else sb.maximum

            if minimum > maximum:
                minimum, maximum = maximum, minimum

            sb.setMinimum(minimum)
            sb.setMaximum(maximum)

            sb.setEnabled(enabled)

    def update_right_cv_info(self):
        """Method to refresh the label with cv infos."""
        l = getattr(self, 'RightCVInfoLabel', None)
        if l is not None:
            lv = self.plsda_model.nr_lv - 1  # because it would start from 1
            rss = np.zeros((self.cv_stats[0][lv].p))
            tss = np.zeros((self.cv_stats[0][lv].p))
            for i in range(len(self.cv_stats)):  # split
                for k in range(self.cv_stats[i][lv].p):
                    print(self.cv_stats[i][lv].rss)
                    rss[k] += self.cv_stats[i][lv].rss[k]
                    tss[k] += self.cv_stats[i][lv].tss[k]

            IO.Log.info("rss {} tss {}".format(rss, tss))
            rmsecv = np.sqrt(rss / self.plsda_model.n / (len(self.cv_stats)))
            r_square = 1 - rss/tss

            text = 'RMSECV:\n{}\n'.format(utility.list_to_string(rmsecv))
            text += 'R² CV:\n{}\n'.format(utility.list_to_string(r_square))
            l.setText(text)

    def update_right_cv_samples_spinbox(self, minimum=None, maximum=None,
                                        enabled=False):
        """Change min, max values and enabled status."""
        sb = getattr(self, 'RightSamplesCVSpinBox', None)
        if sb is not None:
            minimum = minimum if minimum is not None else sb.minimum
            maximum = maximum if maximum is not None else sb.maximum

            if minimum > maximum:
                minimum, maximum = maximum, minimum

            sb.setMinimum(minimum)
            sb.setMaximum(maximum)

            sb.setEnabled(enabled)

    def update_right_cv_splits_spinbox(self, minimum=None, maximum=None,
                                       enabled=False):
        """Change min, max values and enabled status."""
        sb = getattr(self, 'RightSplitsCVSpinBox', None)
        if sb is not None:
            minimum = minimum if minimum is not None else sb.minimum
            maximum = maximum if maximum is not None else sb.maximum

            if minimum > maximum:
                minimum, maximum = maximum, minimum

            sb.setMinimum(minimum)
            sb.setMaximum(maximum)

            sb.setEnabled(enabled)

    def update_right_prediction_info(self):
        """Method to refresh the label with prediction infos."""
        l = getattr(self, 'RightPredictionInfoLabel', None)
        if l is not None:
            text = 'X-Block: {} x {}\n'.format(self.test_set.n,
                                               self.test_set.m)
            text += 'Y-Block: {} x {}\n'.format(self.test_set.n,
                                                self.test_set.p)
            text += 'RMSEP:\n{}\n'.format(
                utility.list_to_string(self.prediction_stats.rmsec))
            text += 'R² Pred:\n{}\n'.format(
                utility.list_to_string(self.prediction_stats.r_squared))
            l.setText(text)

    def clear_plot_lanes_and_show_hints(self):
        """Create a layout with a label to explain how to draw plots."""
        for lane in (Lane.Left, Lane.Central):
            self.reset_widget_and_layout(Widget.Form, lane)
            self.add(QLabel, lane, Column.Both, row=0, name='Hint',
                     text='↑{}↑\n'.format(' ' * 48) +
                     'Several plots are available in the above dropdown menu.'
                     '\n(if you create or load a model before)',
                     label_alignment=Qt.AlignHCenter,
                     policy=Policy.Expanding, size=(170, 520, 3610, 4240))

    def reset_widget_and_layout(self, widget, lane, show=True):
        """Reset the [Left|Central][Form|VBox] Widget and Layout.

           Warning: any previous Widget and Layout will be deleted and lost!

           lane should be of gui.Lane(enum.Enum) type
           widget should be of gui.Widget(enum.Enum) type

           Raises TypeError on bad widget type.

           Returns a tuple with references to the created widget and layout.
        """
        if not isinstance(widget, Widget):
            raise TypeError('widget parameter in reset_widget_and_layout() is '
                            'not instance of Widget ({})'.format(type(widget)))

        # clean up old widget and old layout (if they exists)
        try:
            clear(getattr(self, str(lane) + str(widget) + 'Layout'))
        except AttributeError:
            IO.Log.debug('No {}{}Layout to clean'.format(lane, widget))
        try:
            delete(getattr(self, str(lane) + str(widget) + 'Widget'))
        except AttributeError:
            IO.Log.debug('No {}{}Widget to delete'.format(lane, widget))

        # create new widget
        w = self.set_attr(str(lane) + str(widget), QWidget,
                          parent=self.scroll_area(lane),
                          size=(174, 427, 3611, 4147))
        w.setGeometry(QRect(0, 0, 290, 545))
        w.setLayoutDirection(Qt.LayoutDirectionAuto)

        # create new layout
        if widget is Widget.Form:
            l = self.set_attr(lane, QFormLayout, parent=w, policy=None)
            # Simulate the form layout appearance of QMacStyle
            # (https://doc.qt.io/qt-5/qformlayout.html#details)
            l.setRowWrapPolicy(QFormLayout.DontWrapRows)
            l.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
            l.setFormAlignment(Qt.AlignHCenter | Qt.AlignTop)
            l.setLabelAlignment(Qt.AlignRight)  # as in macOS Aqua guidelines
        else:
            assert widget is Widget.VBox, 'Unexpected widget type in ' \
                                          'reset_widget_and_layout() ' \
                                          '({})'.format(type(widget))
            l = self.set_attr(str(lane), QVBoxLayout, parent=w, policy=None)

            # create back button
            back = self.set_attr(str(lane) + 'Back', QPushButton,
                                 parent=w, policy=Policy.Expanding,
                                 size=(138, 22, 3625, 22))
            back.setText('Back')

            # create new canvas and but keep previous figure (and clear it)
            fig = self.figure(lane)
            fig.clf()
            canvas = Canvas(fig)
            canvas.setSizePolicy(Policy.Expanding, Policy.Expanding)
            canvas.setMinimumSize(QSize(138, 475))
            canvas.setMaximumSize(QSize(3625, 4195))
            setattr(self, str(lane) + 'Canvas', canvas)

            # create toolbar
            toolbar = Toolbar(canvas, parent=w)
            toolbar.setSizePolicy(Policy.Expanding, Policy.Preferred)
            toolbar.setMinimumSize(QSize(138, 30))
            toolbar.setMaximumSize(QSize(3625, 30))
            setattr(self, str(lane) + 'Toolbar', toolbar)

            l.addWidget(toolbar, 0, Qt.AlignHCenter | Qt.AlignTop)
            l.addWidget(canvas)
            l.addWidget(back, 0, Qt.AlignBottom)

        l.setSizeConstraint(QLayout.SetMaximumSize)
        l.setContentsMargins(10, 10, 10, 10)
        l.setSpacing(10)

        if show:
            # any widget that was in scroll_area is lost!
            self.scroll_area(lane).setWidget(w)
            self.scroll_area(lane).setWidgetResizable(True)
            w.show()
        return w, l

    def set_attr(self, name, widget, parent=None, policy=Policy.Expanding,
                 size=None):
        """Wrapper of setattr over self to add a widget."""
        attr_name = str(name) + widget.__name__.lstrip('Q')

        new_widget = widget(parent)
        new_widget.setObjectName(attr_name)
        setattr(self, attr_name, new_widget)

        if policy is not None:
            pol = Policy(policy, policy)
            pol.setHeightForWidth(new_widget.sizePolicy().hasHeightForWidth())
            pol.setHorizontalStretch(0)
            pol.setVerticalStretch(0)
            new_widget.setSizePolicy(pol)
        if size is not None:
            size = tuple(map(int, map(max, size, [0, 0, 0, 0])))
            min_w, min_h, max_w, max_h = size
            min_w, max_w = min(7680, min_w), min(7680, max_w)
            min_h, max_h = min(4320, min_h), min(4320, max_h)
            new_widget.setMinimumSize(QSize(min_w, min_h))
            new_widget.setMaximumSize(QSize(max_w, max_h))
        return new_widget

    def add(self, widget, lane, column, row, name, text=None, word_wrap=True,
            text_format=Qt.AutoText, label_alignment=Qt.AlignRight,
            group_name=None, minimum=1, maximum=99, parent_widget=None,
            parent_type=None, policy=Policy.Preferred,
            size=(70, 25, 170, 520)):
        """Add to the specified lane the widget in (row, column) position."""
        if parent_widget is None:
            if lane == Lane.Right and \
               (parent_type is None or not isinstance(parent_type, Mode)):
                raise TypeError('Please specify at least "parent_widget" or '
                                '"parent_type" (if lane is right) in '
                                'gui.UserInterface.add() method !')
            else:
                parent_widget = self.form_widget(lane, parent_type)
        if lane == Lane.Right:
            parent_name = parent_widget.objectName().lstrip('Right')
            parent_name = parent_name.rstrip('FormWidget')
            if parent_name != 'CV':
                parent_type = getattr(Mode, parent_name)
            else:
                parent_type = Mode.CV
            if not name.startswith(parent_name):
                name += parent_name

        new_widget = self.set_attr(str(lane) + name, widget,
                                   parent=parent_widget,
                                   policy=policy, size=size)

        if widget == QLabel:
            new_widget.setTextFormat(text_format)
            new_widget.setAlignment(label_alignment)
            new_widget.setWordWrap(word_wrap)
            new_widget.setTextInteractionFlags(Qt.TextSelectableByKeyboard
                                               | Qt.TextSelectableByMouse)
        elif widget == QRadioButton:
            if not group_name.endswith('ButtonGroup'):
                group_name += 'ButtonGroup'
            group = getattr(self, group_name, None)
            try:
                if group is not None:
                    group.objectName()
            except RuntimeError as e:
                if 'QButtonGroup has been deleted' in str(e):
                    # group has already been garbage collected
                    IO.Log.debug('Deleted {}'.format(group_name))
                    group = None  # force creation of a new group
                else:
                    raise e

            # create a new group
            if group is None:
                group = QButtonGroup(parent_widget)
                group.setObjectName(group_name)
                setattr(self, group_name, group)
                IO.Log.debug('Created {}ButtonGroup'.format(group_name))

            group.addButton(new_widget)

            # ensure at least one RadioButton is checked
            if group.checkedButton() is None:
                new_widget.setChecked(True)
        elif widget == QSpinBox:
            new_widget.setMinimum(minimum)
            new_widget.setMaximum(maximum)

        if widget in (QCheckBox, QLabel, QPushButton, QRadioButton):
            new_widget.setText(str(text if text is not None else str(name)))

        layout = self.form_layout(lane, parent_type)
        layout.setWidget(row, column.value, new_widget)
        return new_widget

    def cache_current_plot_preferences(self, lane):
        self.clear_cached_plot_preferences(lane)
        cache = getattr(self, str(lane) + 'PlotPreferences')

        for name in ('LVaSpinBox', 'LVbSpinBox', 'LVsSpinBox',
                     'XRadioButton', 'YRadioButton', 'NormalizeCheckBox'):
            if 'SpinBox' in name:
                try:
                    cache[name] = getattr(self, str(lane) + name).value()
                except (AttributeError, RuntimeError):
                    cache.pop(name, None)
            else:
                try:
                    cache[name] = getattr(self, str(lane) + name).isChecked()
                except (AttributeError, RuntimeError):
                    cache.pop(name, None)
            if name in cache:
                IO.Log.debug('Added {}: {} '.format(name, cache[name]) +
                             'to cached plot preferences')

    def clear_cached_plot_preferences(self, lane):
        setattr(self, str(lane) + 'PlotPreferences', dict())

    def set_cached_plot_preferences(self, lane):
        cache = getattr(self, str(lane) + 'PlotPreferences')

        for name in cache:
            if 'SpinBox' in name:  # 'LVaSpinBox', 'LVbSpinBox', 'LVsSpinBox'
                try:
                    getattr(self, str(lane) + name).setValue(cache[name])
                except (AttributeError, RuntimeError) as e:
                    pass  # widget not yet created or already garbage collected
                else:
                    IO.Log.debug('Set {} with cached preference {}'.format(
                        str(lane) + name, str(cache[name])))
            else:  # 'XRadioButton', 'YRadioButton', 'NormalizeCheckBox'
                try:
                    getattr(self, str(lane) + name).setChecked(cache[name])
                except (AttributeError, RuntimeError) as e:
                    pass
                else:
                    IO.Log.debug('Set {} with cached preference {}'.format(
                        str(lane) + name, str(cache[name])))

    def get_cached_plot_preferences(self, lane):
        """Return tuple of cached value or None if absent.

           Order of the tuple fields:
           'LVaSpinBox', 'LVbSpinBox', 'LVsSpinBox', ...
           ... 'NormalizeCheckBox', 'XRadioButton', 'YRadioButton'
        """
        white_list = ('LVaSpinBox', 'LVbSpinBox', 'LVsSpinBox',
                      'NormalizeCheckBox', 'XRadioButton', 'YRadioButton')
        cache = getattr(self, str(lane) + 'PlotPreferences')
        ret = tuple((cache.get(key, None) for key in white_list))
        for k in cache:
            assert k in white_list, 'Unknown cached plot preference: ' + str(k)
        assert len(ret) == 6, 'Cached plot preferences should be 6'
        IO.Log.debug('Cached plot preferences: {}'.format(repr(ret)))
        return ret

    def draw_plot(self, lane, entry, refresh=False):
        if not refresh:
            # Plot button has just been pushed,
            # let's cache immediately preferences
            self.cache_current_plot_preferences(lane)

        self.figure(lane).clear()
        try:
            getattr(self, 'draw_' + entry['method'])(lane, refresh=refresh)
        except Exception as e:
            self.figure(lane).clear()
            # resize the vbox_widget to ensure figure.clear() will be applied
            size = self.vbox_widget(lane).size()
            width, height = size.width(), size.height()
            self.vbox_widget(lane).resize(width - 1, height - 1)
            self.vbox_widget(lane).resize(width, height)

            IO.Log.debug(str(e))
            traceback.print_exc()
            if 'is out of bounds ' in str(e) and '[1:1]' not in str(e) and \
               ('plot.scores()' in str(e) or 'plot.loadings()' in str(e)):
                self.back_button(lane).animateClick(1)
            elif 'is out of bounds [1:1]' not in str(e):
                popup_error(message='An error occured while drawing '
                            '{} plot:\n\n'.format(str(lane).lower()) +
                            '{}'.format(str(e)),
                            parent=self.MainWindow)
        else:
            self.canvas(lane).draw()
            if not refresh:
                self.scroll_area(lane).setWidget(self.vbox_widget(lane))
                self.scroll_area(lane).setWidgetResizable(True)
                self.vbox_widget(lane).show()

    def call_plot_method(self, lane, index=None, text=None):
        if index is None and text is None:
            IO.Log.warning('UserInterface.call_plot_method() got None index '
                           'and None text!')
            return
        for entry in self.drop_down_menu:
            if index == entry['index'] or text == entry['text']:
                # create two new empty layouts
                self.reset_widget_and_layout(Widget.Form, lane, show=True)
                self.reset_widget_and_layout(Widget.VBox, lane, show=False)

                # populate layouts with widget
                try:
                    getattr(self, 'build_' + entry['method'] + '_form')(lane)
                except AttributeError:
                    try:
                        # if there isn't a 'build_' method
                        # maybe there aren't inputs to choose,
                        # let's draw directly
                        self.draw_plot(lane, entry)
                    except AttributeError as f:
                        # unfortunately also the 'draw_' method is missing
                        IO.Log.debug(str(f))
                        popup_error(message=str(f), parent=self.MainWindow)
                    else:
                        # plot drawn with success
                        # no inputs means no need of a back button
                        self.back_button(lane).setVisible(False)
                else:
                    self.set_cached_plot_preferences(lane)
                    self.plot_button(lane).clicked.connect(
                        lambda: self.draw_plot(lane, entry))
                    self.back_button(lane).clicked.connect(
                        lambda: self.call_plot_method(lane, index, text))
                break

    def xy_radio_form(self, lane, group):
        self.add(QRadioButton, lane, Column.Left, row=0, name='X',
                 group_name=str(lane) + group)
        self.add(QRadioButton, lane, Column.Left, row=1, name='Y',
                 group_name=str(lane) + group)
        self.add(QPushButton, lane, Column.Left, row=2, name='Plot',
                 size=(70, 25, 20, 25))

    def xy_radio_ab_spin_form(self, lane, group, add_normalize=False):
        self.add(QRadioButton, lane, Column.Left, row=0, name='X',
                 group_name=str(lane) + group)
        self.add(QRadioButton, lane, Column.Right, row=0, name='Y',
                 group_name=str(lane) + group)

        self.add(QLabel, lane, Column.Left, row=1, name='LVa',
                 text='1st latent variable')
        sb1 = self.add(QSpinBox, lane, Column.Right, row=1, name='LVa',
                       minimum=1, maximum=self.plsda_model.nr_lv)

        self.add(QLabel, lane, Column.Left, row=2, name='LVb',
                 text='2nd latent variable')
        sb2 = self.add(QSpinBox, lane, Column.Right, row=2, name='LVb',
                       minimum=1, maximum=self.plsda_model.nr_lv)
        sb2.setValue(sb1.value() + 1)

        if add_normalize:
            self.add(QCheckBox, lane, Column.Left, row=3, name='Normalize',
                     text='normalize')
        self.add(QPushButton, lane, Column.Right, row=3, name='Plot',
                 size=(70, 25, 20, 25))

        w1 = self.add(QLabel, lane, Column.Left, row=4, name='Warning',
                      text_format=Qt.RichText, label_alignment=Qt.AlignRight,
                      text='<font color="red">Latent <br><br>should</font>')
        w2 = self.add(QLabel, lane, Column.Right, row=4, name='Warning',
                      text_format=Qt.RichText, label_alignment=Qt.AlignLeft,
                      text='<font color="red"> variables<br><br> differ!'
                      '</font>')
        w1.setVisible(False), w2.setVisible(False)
        sb1.valueChanged.connect(
            lambda sb1_value: (w1.setVisible(sb1_value == sb2.value()),
                               w2.setVisible(sb1_value == sb2.value())))
        sb2.valueChanged.connect(
            lambda sb2_value: (w1.setVisible(sb1.value() == sb2_value),
                               w2.setVisible(sb1.value() == sb2_value)))

    def build_scree_plot_form(self, lane):
        self.xy_radio_form(lane, group='ScreePlot')

    def build_explained_variance_plot_form(self, lane):
        self.xy_radio_form(lane, group='ExplainedVariance')

    def build_inner_relations_plot_form(self, lane):
        self.add(QLabel, lane, Column.Left, row=0, name='LVs',
                 text='Latent variable')
        self.add(QSpinBox, lane, Column.Right, row=0, name='LVs',
                 minimum=1, maximum=self.plsda_model.nr_lv)
        self.add(QPushButton, lane, Column.Right, row=1, name='Plot',
                 size=(70, 25, 20, 25))

    def build_scores_plot_form(self, lane):
        self.xy_radio_ab_spin_form(
            lane, group='ScoresPlot', add_normalize=True)

    def build_loadings_plot_form(self, lane):
        self.xy_radio_ab_spin_form(lane, group='LoadingsPlot')

    def build_biplot_plot_form(self, lane):
        self.xy_radio_ab_spin_form(
            lane, group='BiplotPlot', add_normalize=True)

    def build_scores_and_loadings_plot_form(self, lane):
        self.xy_radio_ab_spin_form(
            lane, group='ScoresAndLoadingsPlot', add_normalize=True)

    def build_weights_plot_form(self, lane):
        self.xy_radio_ab_spin_form(lane, group='WeightsPlot')

        # remove unnecessary x, y radio buttons
        x, y = self.x_radio_button(lane), self.y_radio_button(lane)
        x.setVisible(False), y.setVisible(False)
        delete(x), delete(y)

    def build_weights_line_plot_form(self, lane):
        self.add(QLabel, lane, Column.Left, row=0, name='LVs',
                 text='Latent variable')
        self.add(QSpinBox, lane, Column.Right, row=0, name='LVs',
                 minimum=1, maximum=self.plsda_model.nr_lv)
        self.add(QPushButton, lane, Column.Right, row=1, name='Plot',
                 size=(70, 25, 20, 25))

    def draw_scree_plot(self, lane, refresh=False):
        if refresh:
            _, _, _, _, x, y = self.get_cached_plot_preferences(lane)
        else:
            x = self.x_radio_button(lane).isChecked()
            y = self.y_radio_button(lane).isChecked()
        plot.scree(self.figure(lane).add_subplot(111), x=x, y=y)

    def draw_explained_variance_plot(self, lane, refresh=False):
        if refresh:
            _, _, _, _, x, y = self.get_cached_plot_preferences(lane)
        else:
            x = self.x_radio_button(lane).isChecked()
            y = self.y_radio_button(lane).isChecked()
        plot.cumulative_explained_variance(
            self.figure(lane).add_subplot(111), x=x, y=y)

    def draw_inner_relations_plot(self, lane, refresh=False):
        if refresh:
            _, _, lvs, _, _, _ = self.get_cached_plot_preferences(lane)
        else:
            lvs = self.lvs_spin_box(lane).value()
        plot.inner_relations(ax=self.figure(lane).add_subplot(111), num=lvs)

    def draw_scores_plot(self, lane, rows=1, cols=1, pos=1, refresh=False):
        if refresh:
            lva, lvb, _, norm, x, y = self.get_cached_plot_preferences(lane)
        else:
            lva = self.lva_spin_box(lane).value()
            lvb = self.lvb_spin_box(lane).value()
            norm = self.normalize_check_box(lane).checkState() == Qt.Checked
            x = self.x_radio_button(lane).isChecked()
            y = self.y_radio_button(lane).isChecked()
        plot.scores(self.figure(lane).add_subplot(rows, cols, pos),
                    lv_a=lva, lv_b=lvb, normalize=norm, x=x, y=y)

    def draw_loadings_plot(self, lane, rows=1, cols=1, pos=1, refresh=False):
        if refresh:
            lva, lvb, _, _, x, y = self.get_cached_plot_preferences(lane)
        else:
            lva = self.lva_spin_box(lane).value()
            lvb = self.lvb_spin_box(lane).value()
            x = self.x_radio_button(lane).isChecked()
            y = self.y_radio_button(lane).isChecked()
        plot.loadings(self.figure(lane).add_subplot(rows, cols, pos),
                      lv_a=lva, lv_b=lvb, x=x, y=y)

    def draw_biplot_plot(self, lane, refresh=False):
        if refresh:
            lva, lvb, _, norm, x, y = self.get_cached_plot_preferences(lane)
        else:
            lva = self.lva_spin_box(lane).value()
            lvb = self.lvb_spin_box(lane).value()
            norm = self.normalize_check_box(lane).checkState() == Qt.Checked
            x = self.x_radio_button(lane).isChecked()
            y = self.y_radio_button(lane).isChecked()
        plot.biplot(self.figure(lane).add_subplot(111),
                    lv_a=lva, lv_b=lvb, normalize=norm, x=x, y=y)

    def draw_scores_and_loadings_plot(self, lane, refresh=False):
        self.draw_scores_plot(lane, rows=2, cols=1, pos=1, refresh=refresh)
        self.draw_loadings_plot(lane, rows=2, cols=1, pos=2, refresh=refresh)

    def draw_calculated_y_plot(self, lane, refresh=False):
        plot.calculated_y(self.figure(lane).add_subplot(111))

    def draw_predicted_y_real_y_plot(self, lane, refresh=False):
        plot.y_predicted_y_real(self.figure(lane).add_subplot(111))

    def draw_predicted_y_plot(self, lane, refresh=False):
        plot.y_predicted(self.figure(lane).add_subplot(111))

    def draw_t_square_q_plot(self, lane, refresh=False):
        plot.t_square_q(self.figure(lane).add_subplot(111))

    def draw_residuals_leverage_plot(self, lane, refresh=False):
        plot.y_residuals_leverage(self.figure(lane).add_subplot(111))

    def draw_samples_leverage_plot(self, lane, refresh=False):
        plot.leverage(self.figure(lane).add_subplot(111))

    def draw_x_residuals_over_samples_plot(self, lane, refresh=False):
        plot.x_residuals_over_samples(self.figure(lane).add_subplot(111))

    def draw_y_residuals_over_samples_plot(self, lane, refresh=False):
        plot.y_residuals_over_samples(self.figure(lane).add_subplot(111))

    def draw_q_over_leverage_plot(self, lane, refresh=False):
        plot.q_over_leverage(self.figure(lane).add_subplot(111))

    def draw_regression_coefficients_plot(self, lane, refresh=False):
        plot.regression_coefficients(self.figure(lane).add_subplot(111))

    def draw_weights_plot(self, lane, refresh=False):
        if refresh:
            lva, lvb, _, _, _, _ = self.get_cached_plot_preferences(lane)
        else:
            lva = self.lva_spin_box(lane).value()
            lvb = self.lvb_spin_box(lane).value()
        plot.weights(self.figure(lane).add_subplot(111), lv_a=lva, lv_b=lvb)

    def draw_weights_line_plot(self, lane, refresh=False):
        if refresh:
            _, _, lvs, _, _, _ = self.get_cached_plot_preferences(lane)
        else:
            lvs = self.lvs_spin_box(lane).value()
        plot.weights_line(ax=self.figure(lane).add_subplot(111), lv=lvs)

    def draw_data_plot(self, lane, refresh=False):
        plot.data(self.figure(lane).add_subplot(111))

    def draw_rmesec_plot(self, lane, refresh=False):
        plot.rmsec_lv(self.figure(lane).add_subplot(111))

    def draw_rmesecv_plot(self, lane, refresh=False):
        plot.rmsecv_lv(self.figure(lane).add_subplot(111), stats=self.cv_stats)

    def draw_rmesep_plot(self, lane, refresh=False):
        plot.rmsep_lv(self.figure(lane).add_subplot(111))

    def new_model(self):
        """Initialize plsda_model attribute from csv."""
        if not self._replace_current_model():
            return

        input_file = popup_choose_input_file(parent=self.MainWindow,
                                             filter_csv=True)
        if input_file is None:
            return

        try:
            train_set = model.TrainingSet(input_file)
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
                getattr(train_set, method)()
                break
        else:
            IO.Log.debug('CANCEL (not chosen any preprocessing)')

        try:
            plsda_model = model.nipals(train_set.x, train_set.y)
        except Exception as e:
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
            return

        self.train_set = train_set
        self.plsda_model = plsda_model

        IO.Log.debug('Model created correctly')
        self.current_mode = Mode.Model
        self.RightLVsModelSpinBox.setValue(self.plsda_model.nr_lv)
        self.clear_plot_lanes_and_show_hints()

    def save_model(self):
        export_dir = popup_choose_output_directory(parent=self.MainWindow)
        if export_dir is None:
            return
        try:
            IO.dump(export_dir,
                    split=self.right_cv_splits(),
                    sample=self.right_cv_samples())
        except Exception as e:
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
            return
        IO.Log.debug('Model saved correctly')

    def load_model(self):
        """Initialize plsda_model attribute from workspace directory."""
        if not self._replace_current_model():
            return

        ws_dir = popup_choose_input_directory(parent=self.MainWindow)
        if ws_dir is None:
            return

        plsda_model_copy = copy.deepcopy(self.plsda_model)
        train_set_copy = copy.deepcopy(self.train_set)
        try:
            plsda_model, train_set, split, sample = IO.load(ws_dir)
            # Do not change this order, first the training set has to be
            # initialized
            self.train_set = train_set
            # then the model because it depends on the training set to update
            # the statistics in the right lane
            self.plsda_model = plsda_model
        except Exception as e:
            self.plsda_model = plsda_model_copy
            self.train_set = train_set_copy
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
            return

        try:
            self.RightSplitsCVSpinBox.setValue(split)
            self.RightSamplesCVSpinBox.setValue(sample)
            self.cross_validation_wrapper()
        except Exception as e:
            IO.Log.debug(str(e))
            # no need to return here, since model and train_set
            # have been loaded successfully

        IO.Log.debug('Model loaded correctly')
        self.current_mode = Mode.Model
        self.RightLVsModelSpinBox.setValue(self.plsda_model.nr_lv)
        self.clear_plot_lanes_and_show_hints()

    def load_csv_to_predict(self):
        """Initialize test_set and prediction_stats attribute."""
        if self.prediction_stats is not None or self.test_set is not None:
            if not popup_question(message='Are you sure to replace the '
                                          'current test set? ',
                                  parent=self.MainWindow,
                                  title='Replace current test set?'):
                return

        input_file = popup_choose_input_file(parent=self.MainWindow,
                                             filter_csv=True)
        if input_file is None:
            return

        try:
            test_set = model.TestSet(input_file, self.train_set)
        except Exception as e:
            traceback.print_exc()
            popup_error(message='The loaded file is not compatible with the '
                                'curent model',
                        parent=self.MainWindow)
            return

        try:
            stats = model.Statistics(
                y_real=test_set.y,
                y_pred=self.plsda_model.predict(test_set.x))
        except Exception as e:
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
            return
        else:
            self.prediction_stats = stats
            self.test_set = test_set

        IO.Log.debug('TestSet created correctly')
        self.current_mode = Mode.Prediction

    def export_matrices(self):
        all_matrices = (
            ('X', '(n x m', 'matrix of predictors)', 'X'),
            ('T', '(n x m', 'matrix of X scores)', 'T'),
            ('P', '(m x m', 'matrix of X loadings)', 'P'),
            ('E', '(n x m', 'matrix of X residuals)', 'E_x'),
            ('Y', '(n x p', 'matrix of responses)', 'Y'),
            ('U', '(n x m', 'matrix of Y scores)', 'U'),
            ('Q', '(p x m', 'matrix of Y loadings)', 'Q'),
            ('F', '(n x p', 'matrix of Y residuals)', 'E_y'),
            ('Predicted Y in fit', '(n x p', 'matrix)', 'Y_modeled'),
            ('Predicted Y in fit (dummy)', '(n x p', 'matrix of '
             '\u00B1' + '1)', 'Y_modeled_dummy'),
            ('W', '(m x m', 'matrix of PLS weights)', 'W'),
            ('W1', '(m x m', 'matrix)', 'W1'),
            ('B', '(m x p', 'matrix of regression coefficients)', 'B'),
            ('Inner relation', '(m x 1', 'vector of regression coefficients)',
             'b'),
            ('X eigenvalues', '(m x 1', 'vector)', 'x_eigenvalues'),
            ('Y eigenvalues', '(m x 1', 'vector)', 'y_eigenvalues'),
            ('X explained variance', '(m x 1', 'vector)',
             'explained_variance_x'),
            ('Y explained variance', '(m x 1', 'vector)',
             'explained_variance_y'),
            ('X cumulative explained variance', '(m x 1', 'vector)',
             'cumulative_explained_variance_x'),
            ('Y cumulative explained variance', '(m x 1', 'vector)',
             'cumulative_explained_variance_y'),
            ('T²', '(n x 1', 'vector)', 't_square'),
            ('Leverage', '(n x 1', 'vector)', 'leverage'),
            ('Q residuals over X', '(n x 1', 'vector)', 'q_residuals_x'),
            )
        combo, hs, item_list = QComboBox(), '\u200a', list()
        hs_w = combo.fontMetrics().boundingRect(hs).width()
        name_w = max([combo.fontMetrics().boundingRect(name + 10 * hs).width()
                      for name, _, _, _ in all_matrices]) + (hs_w * 0.5)
        size_w = max([combo.fontMetrics().boundingRect(size + 4 * hs).width()
                      for _, size, _, _ in all_matrices]) + (hs_w * 0.5)
        for n, s, desc, _ in all_matrices:
            name = n
            while combo.fontMetrics().boundingRect(name).width() < name_w:
                name += hs
            size = s
            while combo.fontMetrics().boundingRect(size).width() < size_w:
                size += hs
            item_list.append(str(name + size + desc))
        delete(combo)

        ok, index = popup_choose_item('Which matrix would you like to export?',
                                      item_list, parent=self.MainWindow,
                                      title='Export matrix to csv file')
        if not ok:
            return

        path = popup_choose_output_file(self.MainWindow, filter_csv=True)
        if path is None:
            return

        method = str(all_matrices[index][3])
        header = ' '.join(all_matrices[index][:3])
        IO.save_matrix(getattr(self.plsda_model, method), path, header,
                       scientific_notation=True)

    def update_latent_variables_number(self):
        self.right_model_lvs().setEnabled(False)
        self.right_model_change_lvs_button().setEnabled(False)
        try:
            self.plsda_model.nr_lv = self.right_model_lvs().value()
        except AssertionError as e:
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
        else:
            self.update_visible_plots()
        finally:
            self.right_model_lvs().setEnabled(True)
            self.right_model_change_lvs_button().setEnabled(True)

    def update_visible_plots(self):
        for lane in (Lane.Left, Lane.Central):
            # Update LatentVariable maximum in SpinBox
            for name in ('LVa', 'LVb', 'LVs'):
                try:
                    getattr(self, str(lane) + name + 'SpinBox').setMaximum(
                        self.plsda_model.nr_lv)
                except Exception:
                    # probably the SpinBox does not exists yet
                    continue

            # Update visible plots
            try:
                # ensure canvas exists
                canvas = self.canvas(lane)
            except AttributeError:
                continue
            else:
                if canvas.isVisible():
                    cb = getattr(self, str(lane) + 'ComboBox')
                    for entry in self.drop_down_menu:
                        if cb.currentIndex() == entry['index']:
                            self.draw_plot(lane, entry, refresh=True)
                            break

    def cross_validation_wrapper(self):
        split = self.right_cv_splits()
        sample = self.right_cv_samples()
        max_lv = self.plsda_model.max_lv
        try:
            ret = model.cross_validation(self.train_set, split, sample, max_lv)
        except Exception as e:
            IO.Log.debug(str(e))
            popup_error(message=str(e), parent=self.MainWindow)
        else:
            self.cv_stats = ret
            self.update_visible_plots()

    def connect_handlers(self):
        self.NewModelAction.triggered.connect(self.new_model)
        self.SaveModelAction.triggered.connect(self.save_model)
        self.LoadModelAction.triggered.connect(self.load_model)
        self.LoadCsvToPredictAction.triggered.connect(self.load_csv_to_predict)
        self.ExportMatricesAction.triggered.connect(self.export_matrices)
        self.QuitAction.triggered.connect(self.quit)

        self.ModelAction.triggered.connect(
            lambda: setattr(self, 'current_mode', Mode.Model))
        self.CrossValidationAction.triggered.connect(
            lambda: setattr(self, 'current_mode', Mode.CV))
        self.PredictionAction.triggered.connect(
            lambda: setattr(self, 'current_mode', Mode.Prediction))

        self.AboutThisProjectAction.triggered.connect(
            self.about_this_project)
        self.AboutQtAction.triggered.connect(QApplication.aboutQt)

        self.LeftComboBox.currentIndexChanged.connect(
            lambda idx: (self.clear_cached_plot_preferences(Lane.Left),
                         self.call_plot_method(Lane.Left, index=idx)))
        self.CentralComboBox.currentTextChanged.connect(
            lambda txt: (self.clear_cached_plot_preferences(Lane.Central),
                         self.call_plot_method(Lane.Central, text=txt)))

        self.RightLVsModelPushButton.clicked.connect(
            self.update_latent_variables_number)
        self.RightStartCVPushButton.clicked.connect(
            self.cross_validation_wrapper)

    def about_this_project(self):
        dialog = QMessageBox(self.MainWindow)
        dialog.setObjectName('AboutThisProjectMessageBox')
        dialog.setWindowTitle('About this project')
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.setIcon(QMessageBox.NoIcon)
        dialog.setTextFormat(Qt.RichText)  # to use HTML
        dialog.setText(
            '<p align="left">'
            '<big><b>About this project</b></big>'
            '<br><br>'
            'This project has been developed for the <i>Processing of '
            'Scientific Data</i> exam (<a href="https://personale.unimore.it/'
            'rubrica/contenutiAD/cocchi/2016/49729/N0/N0/4569">EDS</a>), at '
            '<i>Physics, Informatics and Mathematics</i> departement (<a '
            'href="http://www.fim.unimore.it/site/en/home.html">FIM</a>) '
            'of <i>University of Modena and Reggio Emilia</i> (<a '
            'href="http://www.unimore.it/en/">UNIMORE</a>) in Italy.'
            '<br><br>'
            'Its main purpose is to conduct a "Partial least squares '
            'Discriminant Analysis" (PLS-DA) on a given dataset.'
            '<br><br>'
            '<br><br>'
            'Copyright (C) 2017  Serena Ziviani, Federico Motta'
            '<br><br>'
            'This program is free software: you can redistribute it and/or '
            'modify it under the terms of the GNU General Public License as '
            'published by the Free Software Foundation, either version 3 of '
            'the License, or any later version.'
            '<br><br>'
            'This program is distributed in the hope that it will be useful, '
            'but WITHOUT ANY WARRANTY; without even the implied warranty of '
            'MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. '
            'See the GNU General Public License for more details.'
            '<br><br>'
            'You should have received a copy of the GNU General Public '
            'License along with this program. '
            'If not, see <a href="http://www.gnu.org/licenses/">'
            'http://www.gnu.org/licenses/</a>.</p>')
        dialog.exec()

    def quit(self, *args):
        """Ask for confirmation with a popup and quit returning 0."""
        if popup_question('Would you really like to quit?', title='Quit',
                          parent=self.MainWindow):
            QCoreApplication.quit()

    def show(self):
        self.MainWindow.show()
