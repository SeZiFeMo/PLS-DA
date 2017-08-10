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

train_set = model.TrainingSet()
train_set.autoscale()
plot.update_global_train_set(train_set)

nipals_model = model.nipals(train_set.x, train_set.y)
plot.update_global_model(nipals_model)
test_set = model.TestSet('datasets/olive_test.csv', train_set)

y_pred = nipals_model.predict(test_set.x)
pred = model.Statistics(test_set.y, y_pred)
plot.update_global_statistics(pred)

results = model.cross_validation(train_set, 4, 6)
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

