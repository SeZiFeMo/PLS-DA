#!/usr/bin/env python3
# coding: utf-8

import IO
import model
import plot
import utility

# Add here every external library used!
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import PyQt5.QtCore as QtCore
    import PyQt5.QtWidgets as QtWidgets
    import scipy
    import sklearn.cross_decomposition
    import yaml
except ImportError as e:
    IO.Log.warning(f'Could not import {e.name} library, '
                   'please install it!')
    exit(1)

if __name__ != '__main__':
    IO.Log.warning('Please do not load that script, run it!')
    exit(1)

utility.check_python_version()

pls_da = model.PLS_DA()

pls_da.preprocess_autoscale()
pls_da.nipals_method()

plt.subplot(2, 2, 1)
# plot.scores_plot(pls_da, 0, 1)
# plot.loadings_plot(pls_da, 0, 1)
# plot.biplot(pls_da, 0, 1)
# plot.explained_variance_plot(pls_da)
# plot.inner_relation_plot(pls_da, 2)
# plot.scree_plot(pls_da, 'y')
plot.y_leverage_plot(pls_da)

plt.subplot(2, 2, 2)
plot.modeled_Y_plot(pls_da)

plt.subplot(2, 2, 3)
plot.data_plot(pls_da, pls_da.allowed_categories)

plt.subplot(2, 2, 4)
plot.d_plot(pls_da)

plt.tight_layout()
plt.show()
