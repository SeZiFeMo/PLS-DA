#!/usr/bin/env python
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
for lib in ('matplotlib.pyplot as plt', 'numpy as np', 'yaml'):
    try:
        exec('import ' + str(lib))
    except ImportError:
        IO.warning('Could not import {} library, '
                   'please install it!'.format(lib))
        exit(1)


pls_da = model.PLS_DA()

pls_da.preprocess_autoscale()
pls_da.nipals_method(nr_lv=4)

pls_da.get_modeled_y()

plt.subplot(2, 1, 1)
plot.scores_plot(pls_da, 0, 1)

plt.subplot(2, 1, 2)
plot.biplot(pls_da, 0, 1)

plt.show()
