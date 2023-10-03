import numpy as np

from typing import Callable


def grad(f: Callable) -> Callable:
    h = 1e-6

    def g(x):
        return (f(x + h) - f(x - h)) / 2 / h

    return g


def grad2(function: Callable) -> Callable:
    h = 1e-6

    def g(*args):
        grads = tuple()
        for i, arg in enumerate(args):
            args_plus = args[:i] + (arg + h,) + args[i+1:]
            args_minus = args[:i] + (arg - h,) + args[i+1:]
            grads += (
                (function(*args_plus) - function(*args_minus)) / 2 / h,
            )
        return grads

    return g


def grad3(function: Callable) -> Callable:
    h = 1e-6

    def g(x: np.ndarray):
        jac = []
        output_shape = function(x).shape
        it = np.nditer(x, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            x_minus = x.copy()
            x_minus[idx] -= h
            x_plus = x.copy()
            x_plus[idx] += h
            jac.append(
                (function(x_plus) - function(x_minus)) / 2 / h
            )
        return np.stack(jac).reshape(output_shape + x.shape)

    return g


def myfun(x):
    return 2 * x


def myfun2(x, y):
    return 2 * x + 3 * y


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def gradsoftmax(x):
    s = softmax(x)
    return -np.outer(s, s) + np.diag(s)


# Simple R^1->R^1 derivative
print(grad2(myfun)(3))
# Multiple arguments R^n->R^1
print(grad2(myfun2)(3, 3))
# Single argument R^n->R^n
x = np.random.uniform(size=(3,))
print(grad3(softmax)(x))
print(gradsoftmax(x))
assert np.allclose(grad3(softmax)(x), gradsoftmax(x)), "Jacobian does not match!"
