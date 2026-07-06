import math
import numpy as np
import matplotlib.pyplot as plt
from value import Value
import graph_trace

def func(x):
    return 3*x**2 - 4*x + 5

# xs = np.arange(-5, 5, 0.25)
# ys = func(xs)
# plt.plot(xs, ys)
# plt.show()

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = f * d; L.label = 'L'
dot = graph_trace.draw_dot(L)
dot.render('graph', view=True)
