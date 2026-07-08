import unittest
from typing import cast
from mlp import MLP
from value import Value
from graph_trace import draw_dot

class TestMLP(unittest.TestCase):
    # python -m unittest test_mlp.TestMLP.test_mlp
    def test_mlp(self):
        x = [2.0, 3.0, -1.0]
        n = MLP(3, [4,4,1])
        print(n(x))

    # python -m unittest test_mlp.TestMLP.test_draw_mlp
    def test_draw_mlp(self):
        x = [2.0, 3.0, -1.0]
        n = MLP(3, [4,4,1])
        print(n(x))
        dot = draw_dot(n(x))
        dot.render('graph', view=True)

    # python -m unittest test_mlp.TestMLP.test_pred
    def test_pred(self):
        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
        ys = [1.0, -1.0, -1.0, 1.0]
        n = MLP(3, [4,4,1])
        ypred = [n(x) for x in xs]
        print(ypred)

    # python -m unittest test_mlp.TestMLP.test_loss
    def test_loss(self):
        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
        ys = [1.0, -1.0, -1.0, 1.0]
        n = MLP(3, [4,4,1])
        ypred = [cast(Value, n(x)) for x in xs]

        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        cast(Value, loss).backward()
        print("grad:",n.layers[0].neurons[0].w[0].grad)
        print("data:",n.layers[0].neurons[0].w[0].data)
        for p in n.parameters():
            p.data += -0.01 * p.grad

        print("data:",n.layers[0].neurons[0].w[0].data)