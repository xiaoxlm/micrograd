import unittest

from value import Value
from graph_trace import draw_dot


class TestGraphTrace(unittest.TestCase):
    #python -m unittest test_graph_trace.TestGraphTrace.test_first_draw_dot
    def test_first_draw_dot(self):
        a = Value(2.0, label='a')
        b = Value(-3.0, label='b')
        c = Value(10.0, label='c')
        e = a * b; e.label = 'e'
        d = e + c; d.label = 'd'
        f = Value(-2.0, label='f')
        L = f * d; L.label = 'L'
        dot = draw_dot(L)
        dot.render('graph', view=True)

    #python -m unittest test_graph_trace.TestGraphTrace.test_draw_dot_with_weight_bias
    def test_draw_dot_with_weight_bias(self):
        # inputs x1, x2
        x1 = Value(2.0, label='x1')
        x2 = Value(0.0, label='x2')
        # weights w1, w2
        w1 = Value(-3.0, label='w1')
        w2 = Value(1.0, label='w2')
        # bias b
        b = Value(6.8813735870195432, label='b')
        # forward pass: x1*w1 + x2*w2 + b
        x1w1 = x1 * w1; x1w1.label = 'x1*w1'
        x2w2 = x2 * w2; x2w2.label = 'x2*w2'
        x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
        n = x1w1x2w2 + b; n.label = 'n'
        o = n.tanh(); o.label = 'o' # using the tanh activation function
        # set grad manually
        o.grad = 1.0
        o._backward()
        n._backward()
        b._backward()
        x1w1x2w2._backward()
        x2w2._backward()
        x1w1._backward()
        
        dot = draw_dot(o) # will show the graph of the neural network
        dot.render('graph', view=True)
    
    #python -m unittest test_graph_trace.TestGraphTrace.test_build_topo
    def test_build_topo(self):
        # inputs x1, x2
        x1 = Value(2.0, label='x1')
        x2 = Value(0.0, label='x2')
        # weights w1, w2
        w1 = Value(-3.0, label='w1')
        w2 = Value(1.0, label='w2')
        # bias b
        b = Value(6.8813735870195432, label='b')
        # forward pass: x1*w1 + x2*w2 + b
        x1w1 = x1 * w1; x1w1.label = 'x1*w1'
        x2w2 = x2 * w2; x2w2.label = 'x2*w2'
        x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
        n = x1w1x2w2 + b; n.label = 'n'
        #----
        # e = (2*n).exp()
        # o = (e-1) / (e+1)
        o = n.tanh(); o.label = 'o' # using the tanh activation function
        # set grad manually
        o.backward()
        # build_topo(o)

        # for node in reversed(topo):
        #     node._backward()
        
        dot = draw_dot(o) # will show the graph of the neural network
        dot.render('graph', view=True)


# python -m unittest test_graph_trace.py
if __name__ == "__main__":
    unittest.main()