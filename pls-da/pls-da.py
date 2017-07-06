#!/usr/bin/env python3
# coding: utf-8

import IO
import model
import plot
import utility

if __name__ != '__main__':
    IO.Log.warning('Please do not load that script, run it!')
    exit(1)

utility.check_python_version()

# Add to the following loop every external library used!
for lib in ('matplotlib.pyplot as plt', 'numpy as np',
            'PyQt5.QtCore as QtCore', 'PyQt5.QtWidgets as QtWidgets', 'scipy',
            'sklearn.cross_decomposition', 'yaml'):
    try:
        exec('import ' + str(lib))
    except ImportError:
        IO.warning('Could not import {} library, '
                   'please install it!'.format(lib))
        exit(1)

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
