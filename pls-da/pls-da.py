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

nipals = model.Nipals(preproc)
nipals.run()

plt.subplot(2, 2, 1)
# plot.scores_plot(nipals, 0, 1)
# plot.loadings_plot(nipals, 0, 1)
# plot.biplot(nipals, 0, 1)
# plot.explained_variance_plot(nipals)
# plot.inner_relation_plot(nipals, 2)
# plot.scree_plot(nipals, 'y')
plot.y_leverage_plot(nipals)

plt.subplot(2, 2, 2)
plot.modeled_Y_plot(nipals)

plt.subplot(2, 2, 3)
plot.data_plot(nipals, model.CATEGORIES)

plt.subplot(2, 2, 4)
plot.d_plot(nipals)

plt.tight_layout()
plt.show()
