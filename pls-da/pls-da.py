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
test_preproc = model.Preprocessing(input_file='datasets/olive_test.csv')
test_x, test_y = preproc.preprocess_test(test_preproc.dataset,
                                         test_preproc.dummy_y)
y_pred = nipals_model.predict(test_x)
pred = model.Statistics(test_y, y_pred)

results = model.cross_validation(preproc, 4, 6)
# for res in results:
#    for stat_id in res:
#        print(res[stat_id].rss)

fig = plt.figure(tight_layout=True)

# plot.scores(ax, 0, 1, x=True)
# plot.loadings(ax, 0, 1, x=True)
# plot.biplot(ax, 0, 1, x=True)
# plot.cumulative_explained_variance(ax, x=True)
# plot.scree(ax, y=True)

plot.weights(fig.add_subplot(2, 2, 1), lv_a=0, lv_b=1)
plot.weights_line(fig.add_subplot(2, 2, 2), lv=0)
plot.weights(fig.add_subplot(2, 2, 3), lv_a=1, lv_b=2)
plot.loadings(fig.add_subplot(2, 2, 4), pc_a=0, pc_b=1, x=True)
plt.show()

