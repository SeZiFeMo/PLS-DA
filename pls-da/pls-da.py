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
test_set = np.arange(10 * nipals_model.m).reshape(10, nipals_model.m)
nipals_model.predict(test_set, None)

results = model.cross_validation(preproc, 4, 6)
for res in results:
    for stat_id in res:
        print(res[stat_id].rss)

fig = plt.figure(tight_layout=True)

# plot.scores(ax, 0, 1, x=True)
# plot.loadings(ax, 0, 1, x=True)
# plot.biplot(ax, 0, 1, x=True)
# plot.cumulative_explained_variance(ax, x=True)
# plot.scree(ax, y=True)

plot.inner_relations(fig.add_subplot(2, 2, 1), num=0)
plot.inner_relations(fig.add_subplot(2, 2, 2), num=1)
plot.inner_relations(fig.add_subplot(2, 2, 3), num=2)
plot.inner_relations(fig.add_subplot(2, 2, 4), num=3)
# plt.show()

