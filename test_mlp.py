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
        for p in n.parameters():
            p.data += -0.01 * p.grad

        print("ypred:",ypred)
    
    # python -m unittest test_mlp.TestMLP.test_standard_training
    def test_standard_training(self):
        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
        ys = [1.0, -1.0, -1.0, 1.0]
        n = MLP(3, [4,4,1])
        ypred = []
        for k in range(20):
            ypred = [cast(Value, n(x)) for x in xs]
            loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

            # backward pass
            for p in n.parameters(): # 非常关键的一步，需要情况 grad。因为如果不清空，会累加梯度。收敛会非常快
                p.grad = 0.0
            cast(Value, loss).backward()

            # update weights
            for p in n.parameters():
                p.data += -0.1 * p.grad # 作用的是 data

            print(f"step {k}, loss: {cast(Value, loss).data}")

        print("ypred:",ypred)
