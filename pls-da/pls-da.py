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
    import sklearn.cross_decomposition as sklCD
    import yaml

except ImportError as e:
    raise SystemExit('Could not import {} library, '.format(e.name)
                     'please install it!')

if __name__ != '__main__':
    raise SystemExit('Please do not load that script, run it!')

utility.check_python_version()

preproc = model.Preprocessing()
preproc.autoscale()
plot.update_global_preproc(preproc)

nipals_model = model.nipals(preproc.dataset, preproc.dummy_y)
plot.update_global_model(nipals_model)

fig = plt.figure(tight_layout=True)

# plot.scores(ax, 0, 1, x=True)
# plot.loadings(ax, 0, 1, x=True)
# plot.biplot(ax, 0, 1, x=True)
# plot.cumulative_explained_variance(ax, x=True)
# plot.inner_relations(ax, num=2)

plot.y_residuals_leverage(fig.add_subplot(2, 3, 1))
plot.calculated_y(fig.add_subplot(2, 3, 2))
plot.scree(fig.add_subplot(2, 3, 3), x=True)
plot.data(fig.add_subplot(2, 3, 4))
plot.regression_coefficients(fig.add_subplot(2, 3, 5))
plot.scree(fig.add_subplot(2, 3, 6), y=True)
plt.show()

