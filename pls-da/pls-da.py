#!/usr/bin/env python3
# coding: utf-8

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

nipals_model = model.nipals(preproc)

plt.subplot(2, 2, 1)
# plot.scores_plot(nipals_model, 0, 1)
# plot.loadings_plot(nipals_model, 0, 1)
# plot.biplot(nipals_model, 0, 1)
# plot.explained_variance_plot(nipals_model)
# plot.inner_relation_plot(nipals_model, 2)
# plot.scree_plot(nipals_model, 'y')
plot.y_leverage_plot(nipals_model)

plt.subplot(2, 2, 2)
plot.modeled_Y_plot(nipals_model)

plt.subplot(2, 2, 3)
plot.data_plot(nipals_model, model.CATEGORIES)

plt.subplot(2, 2, 4)
plot.d_plot(nipals_model)

plt.tight_layout()
plt.show()
