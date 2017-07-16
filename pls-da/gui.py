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


class Column(enum.Enum):
    """Enumerate to identify the columns of the QtWidgets.QFormLayout."""

    Left = QtWidgets.QFormLayout.LabelRole
    Right = QtWidgets.QFormLayout.FieldRole
    Both = QtWidgets.QFormLayout.SpanningRole


class Lane(enum.Enum):
    """Enumerate to identify the lanes of the gui which can contain plots."""

    Left = 'Left'
    Central = 'Central'
    Right = 'Right'


class UserInterface(object):

    def __init__(self, main_window_title):
        main_window = self.set_attr('', 'QMainWindow', policy='Expanding',
                                    min_size=(800, 600), max_size=(7680, 4320))
        main_window.setUnifiedTitleAndToolBarOnMac(True)
        main_window.setWindowTitle(main_window_title)
        main_window.setEnabled(True)
        main_window.resize(800, 600)

        main_widget = self.set_attr(
                'Main', 'QWidget', parent=main_window, policy='Preferred',
                min_size=(800, 600), max_size=(7680, 4300))

        main_layout = self.set_attr('Main', 'QGridLayout', parent=main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_window.setCentralWidget(main_widget)

        main_splitter = self.set_attr(
                'Main', 'QSplitter', parent=main_widget, policy='Expanding',
                min_size=(800, 600), max_size=(7680, 4300))
        main_layout.addWidget(main_splitter, 0, 0, 1, 1)
        main_splitter.setOrientation(QtCore.Qt.Horizontal)
        main_splitter.setHandleWidth(3)

        for lane in (Lane.Left, Lane.Central):
            parent = self.set_attr(lane.value, 'QWidget', parent=main_splitter,
                                   min_size=(200, 580), max_size=(3637, 4300))

            layout = self.set_attr(lane.value, 'QGridLayout', parent=parent)
            layout.setContentsMargins(3, 3, 3, 3)
            layout.setSpacing(5)

            drop_down = self.set_attr(
                    lane.value, 'QComboBox', parent=parent, policy='Expanding',
                    min_size=(194, 22), max_size=(3631, 22))
            layout.addWidget(drop_down, 0, 0, 1, 1)
            # Previously in retranslateUi()
            for entry in self.drop_down_menu:
                drop_down.addItem(entry['text'])
            drop_down.setCurrentIndex(-1)

            scroll_area = self.set_attr(
                    lane.value, 'QScrollArea', parent=parent,
                    min_size=(194, 547), max_size=(3631, 4267))
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area, 1, 0, 1, 1)

            scroll_area_widget = self.set_attr(
                    lane.value + 'ScrollArea', 'QWidget', policy='Expanding',
                    min_size=(174, 427), max_size=(3611, 4147))
            scroll_area.setWidget(scroll_area_widget)
            scroll_area_widget.setGeometry(QtCore.QRect(0, 0, 290, 565))
            scroll_area_widget.setLayoutDirection(QtCore.Qt.LeftToRight)

            form_layout = self.set_attr(lane.value + 'Plot', 'QFormLayout',
                                        parent=scroll_area_widget)
            form_layout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
            form_layout.setFieldGrowthPolicy(
                    QtWidgets.QFormLayout.ExpandingFieldsGrow)
            form_layout.setLabelAlignment(QtCore.Qt.AlignCenter)
            form_layout.setFormAlignment(QtCore.Qt.AlignHCenter
                                         | QtCore.Qt.AlignTop)
            form_layout.setContentsMargins(10, 10, 10, 10)
            form_layout.setSpacing(10)

            self.add('QLabel', lane, Column.Left, row=0, name='LVs',
                     text='Latent Variables')
            self.add('QSpinBox', lane, Column.Right, row=0, name='LVs',)
            self.add('QRadioButton', lane, Column.Left, row=1, name='X',
                     group_name=lane.value + 'ButtonGroup')
            self.add('QSpinBox', lane, Column.Right, row=1, name='X')
            self.add('QRadioButton', lane, Column.Left, row=2, name='Y',
                     group_name=lane.value + 'ButtonGroup')
            self.add('QSpinBox', lane, Column.Right, row=2, name='Y')
            self.add('QPushButton', lane, Column.Left, row=3, name='Back')
            self.add('QPushButton', lane, Column.Right, row=3, name='Plot')

        lane = Lane.Right
        parent = self.set_attr(
                lane.value, 'QWidget', parent=main_splitter,
                policy='Expanding', min_size=(150, 580), max_size=(400, 4300))

        layout = self.set_attr(lane.value, 'QGridLayout', parent=parent)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(5)

        current_mode_label = self.set_attr(
                'CurrentMode', 'QLabel', parent=parent, policy='Expanding',
                min_size=(144, 22), max_size=(394, 22))
        current_mode_label.setLineWidth(1)
        current_mode_label.setTextFormat(QtCore.Qt.AutoText)
        current_mode_label.setAlignment(QtCore.Qt.AlignCenter)
        current_mode_label.setFrameShadow(QtWidgets.QFrame.Plain)
        current_mode_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        layout.addWidget(current_mode_label, 0, 0, 1, 1)

        scroll_area = self.set_attr('Right', 'QScrollArea', parent=parent,
                                    min_size=(144, 547), max_size=(394, 4272))
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area, 1, 0, 1, 1)

        scroll_area_widget = self.set_attr(
                'RightScrollArea', 'QWidget', policy='Expanding',
                min_size=(138, 534), max_size=(388, 4259))
        scroll_area.setWidget(scroll_area_widget)
        scroll_area_widget.setGeometry(QtCore.QRect(0, 0, 189, 565))

        details_layout = self.set_attr('RightDetails', 'QGridLayout',
                                       parent=self.RightScrollAreaWidget)
        details_layout.setContentsMargins(3, 3, 3, 3)
        details_layout.setSpacing(5)

        details_label = self.set_attr(
                'Details', 'QLabel', parent=scroll_area_widget,
                policy='Expanding', min_size=(138, 534), max_size=(388, 4259))
        details_layout.addWidget(self.DetailsLabel, 0, 0, 1, 1)
        details_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        details_label.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByKeyboard
                | QtCore.Qt.TextSelectableByMouse)
        details_label.setText("Details")
        details_label.setWordWrap(True)

        # MenuBar
        top_menu_bar = self.set_attr('Top', 'QMenuBar', parent=main_window,
                                     min_size=(800, 20), max_size=(7680, 20))
        top_menu_bar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        main_window.setMenuBar(top_menu_bar)

        for menu in self.menu_bar:
            self.set_attr(menu['name'], 'QMenu', parent=top_menu_bar,
                          min_size=(100, 20), max_size=(960, 4300))
            menu_obj = getattr(self, menu['name'] + 'Menu')
            top_menu_bar.addAction(menu_obj.menuAction())
            menu_obj.setTitle(menu['title'])

            for action in self.menu_action(menu['name']):
                self.set_attr(action['name'], 'QAction', parent=menu_obj)
                action_obj = getattr(self, action['name'] + 'Action')
                if action['shortcut'] is None:
                    action_obj.setSeparator(True)
                    action_obj.setText('')
                else:
                    action_obj.setShortcut(action['shortcut'])
                    action_obj.setText(action['text'])
                menu_obj.addAction(action_obj)

        QtCore.QMetaObject.connectSlotsByName(main_window)
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
            yield {'name': name.replace(' ', ''), 'title': '&' + name}

    def menu_action(self, menu):
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

        self.ModelAction.setEnabled(model_flag)
        self.CrossValidationAction.setEnabled(cv_flag)
        self.PredictionAction.setEnabled(pred_flag)

        if self._current_mode == 'start':
            self.SaveModelAction.setEnabled(False)
            self.ExportMatricesAction.setEnabled(False)
        else:
            self.SaveModelAction.setEnabled(True)
            self.ExportMatricesAction.setEnabled(True)


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

    def set_attr(self, name, widget, parent=None, lane=None,
                 policy=None, min_size=None, max_size=None):
        """Wrapper of qt5 module function and setattr over self."""
        attr_name = lane.value if lane is not None else ''
        attr_name += str(name) + widget.lstrip('Q')

        new_widget = new_qt(widget, attr_name, parent=parent)
        setattr(self, attr_name, new_widget)

        if policy is not None:
            tmp = QtWidgets.QSizePolicy(getattr(QtWidgets.QSizePolicy, policy),
                                        getattr(QtWidgets.QSizePolicy, policy))
            tmp.setHeightForWidth(new_widget.sizePolicy().hasHeightForWidth())
            tmp.setHorizontalStretch(0)
            tmp.setVerticalStretch(0)
            new_widget.setSizePolicy(tmp)
        if min_size is not None and max_size is not None:
            for sizes, attr in ((min_size, 'setMinimumSize'),
                                (max_size, 'setMaximumSize')):
                if sizes is not None and \
                   (isinstance(sizes, tuple) or isinstance(sizes, list)) and \
                   len(sizes) == 2:
                    sizes = tuple(map(int, sizes))
                    if sizes[0] in range(0, 7681) and \
                       sizes[1] in range(0, 4321):
                        getattr(new_widget, attr)(QtCore.QSize(*sizes))
                    else:
                        func = '.'.join(new_widget.objectName(), attr)
                        func = func.lstrip('.')
                        IO.Log.warning('{}({},{}) '.format(func, *sizes)
                                       'failed! Width not in [0; 7680] '
                                       'or height not in [0; 4320]')
        return new_widget

    def add(self, widget, lane, column, row, name, text=None, word_wrap=True,
            text_format=QtCore.Qt.AutoText, alignment=QtCore.Qt.AlignLeft,
            group_name=None, minimum=1, maximum=99):
        """Add to the specified lane the widget in (row, column) position."""
        parent_widget = getattr(self, lane.value + 'ScrollAreaWidget')
        new_widget = self.set_attr(name, widget, parent=parent_widget,
                                   lane=lane, policy='Preferred',
                                   min_size=(70, 22), max_size=(1310, 170))

        if widget == 'QLabel':
            new_widget.setTextFormat(text_format)
            new_widget.setAlignment(alignment)
            new_widget.setWordWrap(word_wrap)
        elif widget == 'QRadioButton':
            group = getattr(self, group_name, None)
            if group is None:
                group = new_qt('QButtonGroup', group_name, parent=parent_widget)
                setattr(self, group_name, group)
            group.addButton(new_widget)
        elif widget == 'QSpinBox':
            new_widget.setMinimum(minimum)
            new_widget.setMaximum(maximum)

        if widget in ('QLabel', 'QPushButton', 'QRadioButton'):
            new_widget.setText(str(text if text is not None else str(name)))

        layout = getattr(self, lane.value + 'PlotFormLayout')
        layout.setWidget(row, column.value, new_widget)
        return new_widget

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

        self.AboutQtAction.triggered.connect(QtWidgets.QApplication.aboutQt)

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
