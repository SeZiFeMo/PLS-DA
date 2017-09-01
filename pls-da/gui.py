#!/usr/bin/env python3
# coding: utf-8

import copy
import enum
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as Toolbar
from PyQt5.QtCore import QCoreApplication, QMetaObject, QRect, QSize, Qt
from PyQt5.QtWidgets import (QAction, QApplication, QButtonGroup, QCheckBox,
                             QComboBox, QDialog, QFileDialog, QFormLayout,
                             QGridLayout, QFrame, QInputDialog, QLabel,
                             QLayout, QMainWindow, QMenu, QMenuBar,
                             QMessageBox, QPushButton, QRadioButton,
                             QScrollArea, QSizePolicy as Policy, QSpinBox,
                             QSplitter, QVBoxLayout, QWidget)
import sys

import IO
import model
import plot
import utility


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
        dialog.setFileMode(QFileDialog.ExistingFile)
        if filter_csv:
            dialog.setNameFilter("Comma-separated values files (*.csv *.txt)")
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
                'CurrentMode', QLabel, parent=parent,
                size=(144, 22, 394, 22))
        current_mode_label.setLineWidth(1)
        current_mode_label.setTextFormat(Qt.AutoText)
        current_mode_label.setAlignment(Qt.AlignCenter)
        current_mode_label.setFrameShadow(QFrame.Plain)
        current_mode_label.setFrameShape(QFrame.StyledPanel)
        layout.addWidget(current_mode_label, 0, 0, 1, 1)

        scroll_area = self.set_attr(lane, QScrollArea, parent=parent,
                                    policy=None, size=(144, 547, 394, 4272))
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area, 1, 0, 1, 1)

        details_grid_widget = self.set_attr(str(lane) + 'DetailsGrid', QWidget,
                                            size=(138, 534, 388, 4259))
        scroll_area.setWidget(details_grid_widget)
        details_grid_widget.setGeometry(QRect(0, 0, 189, 565))

        details_layout = self.set_attr(str(lane) + 'Details', QGridLayout,
                                       parent=details_grid_widget, policy=None)
        details_layout.setContentsMargins(3, 3, 3, 3)
        details_layout.setSpacing(5)

        details_label = self.set_attr(
                'Details', QLabel, parent=details_grid_widget,
                size=(138, 534, 388, 4259))
        details_layout.addWidget(details_label, 0, 0, 1, 1)
        details_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        details_label.setTextInteractionFlags(Qt.TextSelectableByKeyboard
                                              | Qt.TextSelectableByMouse)
        details_label.setWordWrap(True)
        self.details_label = 'Details'

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

        self.plsda_model = None
        self.current_mode = 'start'

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
               ('Predicted Y', 'predicted_y'),
               ('T² – Q', 't_square_q'),
               ('Residuals – Leverage', 'residuals_leverage'),
               ('Regression coefficients', 'regression_coefficients'))
        for index, (text, method) in enumerate(tmp):
            yield {'index': index, 'text': text, 'method': method + '_plot'}

    @property
    def menu_bar(self):
        """Return a generator iterator over menu items properties."""
        for name in ('Options', 'Change Mode', 'About'):
            yield {'name': name.replace(' ', ''), 'title': '&' + name}

    def menu_action(self, menu):
        """Return a generator iterator over action items properties."""
        menu = menu.lstrip('Menu').replace(' ', '').replace('&', '')
        if menu == 'Options':
            tmp = (('&New model', 'Ctrl+N'),
                   ('&Save model', 'Ctrl+S'),
                   ('&Load model', 'Ctrl+L'),
                   ('1_ Separator', None),
                   ('&Export matrices', 'Ctrl+E'),
                   ('2_ Separator', None),
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
            yield {'name': ''.join(l).replace('sv', 'sV'),
                   'text': text, 'shortcut': shortcut}

    def back_button(self, lane):
        return getattr(self, str(lane) + 'BackPushButton')

    def canvas(self, lane):
        return getattr(self, str(lane) + 'Canvas')

    def figure(self, lane):
        return getattr(self, str(lane) + 'Figure')

    def form_layout(self, lane):
        return getattr(self, str(lane) + 'FormLayout')

    def form_widget(self, lane):
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
        """Get value of CurrentModeLabel."""
        return self.CurrentModeLabel.text()

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
        self.CurrentModeLabel.setText(
                self._current_mode.capitalize().replace('sv', 'sV') + ' mode')

        model_flag, cv_flag, pred_flag = False, False, False
        if self._current_mode == 'crossvalidation':
            model_flag, pred_flag = True, True
        elif self._current_mode == 'model':
            cv_flag, pred_flag = True, True
        elif self._current_mode == 'prediction':
            model_flag, cv_flag = True, True

        self.ModelAction.setEnabled(model_flag)
        self.CrossValidationAction.setEnabled(cv_flag)
        self.PredictionAction.setEnabled(pred_flag)

        if self._current_mode == 'start':
            self.SaveModelAction.setEnabled(False)
            self.ExportMatricesAction.setEnabled(False)

            self.LeftComboBox.setEnabled(False)
            self.CentralComboBox.setEnabled(False)

            for lane in (Lane.Left, Lane.Central):
                self.reset_widget_and_layout(Widget.Form, lane)
                self.add(QLabel, lane, Column.Both, row=0, name='Hint',
                         text='Several plots are available in the above '
                              'dropdown menu.\n'
                              '(if you create or load a model before)',
                         label_alignment=Qt.AlignHCenter,
                         policy=Policy.Expanding, size=(170, 520, 3610, 4240))
        else:
            self.SaveModelAction.setEnabled(True)
            self.ExportMatricesAction.setEnabled(True)

            self.LeftComboBox.setEnabled(True)
            self.CentralComboBox.setEnabled(True)


    @property
    def details_label(self):
        """Get text of DetailsLabel."""
        return self.DetailsLabel.text()

    @details_label.setter
    def details_label(self, value):
        """Set text of DetailsLabel."""
        return self.DetailsLabel.setText(str(value))

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
            assert widget is Widget.VBox, "Unexpected widget type in " \
                                          "reset_widget_and_layout() " \
                                          "({})".format(type(widget))
            l = self.set_attr(str(lane), QVBoxLayout, parent=w, policy=None)

            # create back button
            back = self.set_attr(str(lane) + 'Back', QPushButton,
                                 parent=w, policy=Policy.Expanding,
                                 size=(138, 22, 3625, 22))
            back.setText('Back')

            # create canvas and its figure
            fig = plt.figure(tight_layout=True)
            setattr(self, str(lane) + 'Figure', fig)
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
            group_name=None, minimum=1, maximum=99, policy=Policy.Preferred,
            size=(70, 25, 170, 520)):
        """Add to the specified lane the widget in (row, column) position."""
        parent_widget = self.form_widget(lane)
        new_widget = self.set_attr(str(lane) + name, widget,
                                   parent=parent_widget,
                                   policy=policy, size=size)

        if widget == QLabel:
            new_widget.setTextFormat(text_format)
            new_widget.setAlignment(label_alignment)
            new_widget.setWordWrap(word_wrap)
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

        layout = self.form_layout(lane)
        layout.setWidget(row, column.value, new_widget)
        return new_widget

    def draw_plot(self, lane, entry):
        try:
            getattr(self, 'draw_' + entry['method'])(lane)
        except ValueError as e:
            popup_error(message=str(e), parent=self.MainWindow)
            self.figure(lane).clear()

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
                getattr(self, 'build_' + entry['method'] + '_form')(lane)

                # connect events
                self.plot_button(lane).clicked.connect(
                    lambda: (self.figure(lane).clear(),
                             self.draw_plot(lane, entry),
                             self.canvas(lane).draw(),
                             self.scroll_area(lane).setWidget(
                                 self.vbox_widget(lane)),
                             self.scroll_area(lane).setWidgetResizable(True),
                             self.vbox_widget(lane).show()))
                self.back_button(lane).clicked.connect(
                    lambda: self.call_plot_method(lane, index, text))
                break

    def xy_radio_form(self, lane, group):
        self.add(QRadioButton, lane, Column.Left, row=0, name='X',
                 group_name=str(lane) + group)
        self.add(QRadioButton, lane, Column.Left, row=1, name='Y',
                 group_name=str(lane) + group)
        self.add(QPushButton, lane, Column.Left, row=2, name='Plot')

    def xy_radio_ab_spin_form(self, lane, group):
        self.add(QRadioButton, lane, Column.Left, row=0, name='X',
                 group_name=str(lane) + group)
        self.add(QRadioButton, lane, Column.Right, row=0, name='Y',
                 group_name=str(lane) + group)

        self.add(QLabel, lane, Column.Left, row=1, name='LVa',
                 text='1st latent variable')
        self.add(QSpinBox, lane, Column.Right, row=1, name='LVa',
                 minimum=0, maximum=self.plsda_model.max_lv - 1)

        self.add(QLabel, lane, Column.Left, row=2, name='LVb',
                 text='2nd latent variable')
        self.add(QSpinBox, lane, Column.Right, row=2, name='LVb',
                 minimum=0, maximum=self.plsda_model.max_lv - 1)

    def only_plot_button_form(self, lane, size=(170, 25, 3610, 25)):
        self.add(QPushButton, lane, Column.Left, row=0, name='Plot', size=size)

    def build_scree_plot_form(self, lane):
        self.xy_radio_form(lane, group='ScreePlot')

    def build_explained_variance_plot_form(self, lane):
        self.xy_radio_form(lane, group='ExplainedVariance')

    def build_inner_relations_plot_form(self, lane):
        self.add(QLabel, lane, Column.Left, row=0, name='LVs',
                 text='Latent variable')
        self.add(QSpinBox, lane, Column.Right, row=0, name='LVs',
                 minimum=0, maximum=self.plsda_model.max_lv - 1)
        self.add(QPushButton, lane, Column.Right, row=1, name='Plot')

    def build_scores_plot_form(self, lane):
        self.xy_radio_ab_spin_form(lane, group='ScoresPlot')
        self.add(QCheckBox, lane, Column.Left, row=3, name='Normalize',
                 text='normalize')
        self.add(QPushButton, lane, Column.Right, row=3, name='Plot')

    def build_loadings_plot_form(self, lane):
        self.xy_radio_ab_spin_form(lane, group='LoadingsPlot')
        self.add(QPushButton, lane, Column.Right, row=3, name='Plot')

    def build_biplot_plot_form(self, lane):
        self.xy_radio_ab_spin_form(lane, group='BiplotPlot')
        self.add(QCheckBox, lane, Column.Left, row=3, name='Normalize',
                 text='normalize')
        self.add(QPushButton, lane, Column.Right, row=3, name='Plot')

    def build_scores_and_loadings_plot_form(self, lane):
        self.xy_radio_ab_spin_form(lane, group='ScoresAndLoadingsPlot')
        self.add(QCheckBox, lane, Column.Left, row=3, name='Normalize',
                 text='normalize')
        self.add(QPushButton, lane, Column.Right, row=3, name='Plot')

    def build_calculated_y_plot_form(self, lane):
        self.only_plot_button_form(lane)

    def build_predicted_y_plot_form(self, lane):
        self.only_plot_button_form(lane)

    def build_t_square_q_plot_form(self, lane):
        self.only_plot_button_form(lane)

    def build_residuals_leverage_plot_form(self, lane):
        self.only_plot_button_form(lane)

    def build_regression_coefficients_plot_form(self, lane):
        self.only_plot_button_form(lane)

    def draw_scree_plot(self, lane, rows=1, cols=1, pos=1):
        plot.scree(self.figure(lane).add_subplot(rows, cols, pos),
                   x=self.x_radio_button(lane).isChecked(),
                   y=self.y_radio_button(lane).isChecked())

    def draw_explained_variance_plot(self, lane, rows=1, cols=1, pos=1):
        plot.cumulative_explained_variance(
            self.figure(lane).add_subplot(rows, cols, pos),
            x=self.x_radio_button(lane).isChecked(),
            y=self.y_radio_button(lane).isChecked())

    def draw_inner_relations_plot(self, lane, rows=1, cols=1, pos=1):
        plot.inner_relations(ax=self.figure(lane).add_subplot(rows, cols, pos),
                             num=self.lvs_spin_box(lane).value())

    def draw_scores_plot(self, lane, rows=1, cols=1, pos=1):
        normalize = self.normalize_check_box(lane).checkState() == Qt.Checked
        plot.scores(self.figure(lane).add_subplot(rows, cols, pos),
                    lv_a=self.lva_spin_box(lane).value(),
                    lv_b=self.lvb_spin_box(lane).value(),
                    x=self.x_radio_button(lane).isChecked(),
                    y=self.y_radio_button(lane).isChecked(),
                    normalize=normalize)

    def draw_loadings_plot(self, lane, rows=1, cols=1, pos=1):
        plot.loadings(self.figure(lane).add_subplot(rows, cols, pos),
                      lv_a=self.lva_spin_box(lane).value(),
                      lv_b=self.lvb_spin_box(lane).value(),
                      x=self.x_radio_button(lane).isChecked(),
                      y=self.y_radio_button(lane).isChecked())

    def draw_biplot_plot(self, lane, rows=1, cols=1, pos=1):
        normalize = self.normalize_check_box(lane).checkState() == Qt.Checked
        plot.biplot(self.figure(lane).add_subplot(rows, cols, pos),
                    lv_a=self.lva_spin_box(lane).value(),
                    lv_b=self.lvb_spin_box(lane).value(),
                    x=self.x_radio_button(lane).isChecked(),
                    y=self.y_radio_button(lane).isChecked(),
                    normalize=normalize)

    def draw_scores_and_loadings_plot(self, lane):
        self.draw_scores_plot(lane, rows=2, cols=1, pos=1)
        self.draw_loadings_plot(lane, rows=2, cols=1, pos=2)

    def draw_calculated_y_plot(self, lane, rows=1, cols=1, pos=1):
        plot.calculated_y(self.figure(lane).add_subplot(rows, cols, pos))

    def draw_predicted_y_plot(self, lane, rows=1, cols=1, pos=1):
        plot.y_predicted(self.figure(lane).add_subplot(rows, cols, pos))

    def draw_t_square_q_plot(self, lane, rows=1, cols=1, pos=1):
        plot.t_square_q(self.figure(lane).add_subplot(rows, cols, pos))

    def draw_residuals_leverage_plot(self, lane, rows=1, cols=1, pos=1):
        plot.y_residuals_leverage(
            self.figure(lane).add_subplot(rows, cols, pos))

    def draw_regression_coefficients_plot(self, lane, rows=1, cols=1, pos=1):
        plot.regression_coefficients(
            self.figure(lane).add_subplot(rows, cols, pos))

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

        self.train_set, self.plsda_model = train_set, plsda_model

        plot.update_global_train_set(self.train_set)
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
        self.NewModelAction.triggered.connect(self.new_model)
        self.SaveModelAction.triggered.connect(self.save_model)
        self.LoadModelAction.triggered.connect(self.load_model)

        self.ExportMatricesAction.triggered.connect(
                lambda: popup_error('exception.NotImplementedError', parent=self.MainWindow))

        self.QuitAction.triggered.connect(self.quit)

        self.ModelAction.triggered.connect(
                lambda: setattr(self, 'current_mode', 'model'))
        self.CrossValidationAction.triggered.connect(
                lambda: setattr(self, 'current_mode', 'cv'))
        self.PredictionAction.triggered.connect(
                lambda: setattr(self, 'current_mode', 'prediction'))

        self.AboutQtAction.triggered.connect(QApplication.aboutQt)

        self.LeftComboBox.currentIndexChanged.connect(
                lambda idx: self.call_plot_method(Lane.Left, index=idx))
        self.CentralComboBox.currentTextChanged.connect(
                lambda txt: self.call_plot_method(Lane.Central, text=txt))

    def quit(self):
        """Ask for confirmation with a popup and quit returning 0."""
        if popup_question('Would you really like to quit?', title='Quit',
                          parent=self.MainWindow):
            QCoreApplication.quit()

    def show(self):
        self.MainWindow.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UserInterface('PLS-DA')
    ui.show()
    if utility.CLI.args().verbose:
        ui.MainWindow.dumpObjectTree()
        #  ui.MainWindow.dumpObjectInfo()
    sys.exit(app.exec_())
