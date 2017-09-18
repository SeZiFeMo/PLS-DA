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


if __name__ != '__main__':
    raise SystemExit('Please do not load that script, run it!')

for lib in ('matplotlib', 'numpy', 'PyQt5', 'scipy', 'yaml'):
    try:
        exec('import ' + str(lib))
    except ImportError:
        raise SystemExit('Could not import {} library, '.format(lib) +
                         'please install it!')


from PyQt5.QtCore import QCoreApplication, QTimer
from PyQt5.QtWidgets import QApplication
import signal
import sys

import gui
import utility

# check python version
if sys.version_info < (3,):
    major, minor, *__ = sys.version_info
    raise SystemExit('WARNING: You are using the Python interpreter {}.{}.\n'
                     'Please use at least Python version 3!'.format(major,
                                                                    minor))

# Create graphical environment
application = QApplication(sys.argv)
user_interface = gui.UserInterface('PLS-DA')

# catch Ctrl+C or INTERRUPT signal
signal.signal(signal.SIGINT, user_interface.quit)  # asks confirmation

# catch TERMINATION signal
signal.signal(signal.SIGTERM,
              lambda *args: QCoreApplication.quit())  # quit immediately

# Create a timer to let the python interpreter run ...
timer = QTimer()
timer.start(500)  # ... every 500 milliseconds ...
timer.timeout.connect(lambda: None)  # ... and just do nothing

if utility.CLI.args().verbose:
    user_interface.MainWindow.dumpObjectTree()

# Start Qt event loop
user_interface.show()
sys.exit(application.exec_())
