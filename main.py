#!/usr/bin/env python
# coding: utf-8

import PKG.io
import PKG.model
import PKG.plot
import PKG.utility

if __name__ != '__main__':
    PKG.io.Log.warning('Please do not load that script, run it!')
    exit(1)

PKG.utility.check_python_version()

# Add to the following loop every external library used!
for lib in ('matplotlib.pyplot as plt', 'numpy as np', 'yaml'):
    try:
        exec('import ' + str(lib))
    except ImportError:
        PKG.io.warning('Could not import {} library, '
                       'please install it!'.format(lib))
        exit(1)


pls_da = PKG.model.PLS_DA()

pls_da.get_dummy_variables()
pls_da.preprocess_autoscale()
pls_da.nipals_method(nr_lv=4)

plt.subplot(2, 1, 1)
PKG.plot.scores_plot(pls_da, 0, 1)

plt.subplot(2, 1, 2)
PKG.plot.loadings_plot(pls_da, 0, 1)

plt.show()
