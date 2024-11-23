import jax.numpy as jnp
from jax import value_and_grad
import numpy as np
from mnist import mnist

w1 = jnp.array(np.random.normal(size=(784, 128)).astype(np.float32)) * 0.01
b1 = np.zeros((128,), dtype=np.float32)

w2 = jnp.array(np.random.normal(size=(128, 10)).astype(np.float32)) * 0.01
b2 = jnp.zeros((10,), dtype=np.float32)

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def dense(x, w, b):
    return sigmoid(jnp.dot(x, w) + b)

def mse(y_true, y_pred):
    return jnp.mean(jnp.square(y_true - y_pred))

def f(w1, b1, w2, b2, y, x):
    h = dense(x, w1, b1)
    y_pred = dense(h, w2, b2)
    loss = mse(y, y_pred)
    return loss

lr = 0.0001

def optim(w1, b1, w2, b2, y, x):
    v, (w1_grad, b1_grad, w2_grad, b2_grad) = value_and_grad(f, argnums=range(4))(w1, b1, w2, b2, y, x)
    w1 -= lr * w1_grad
    b1 -= lr * b1_grad
    w2 -= lr * w2_grad
    b2 -= lr * b2_grad
    return w1, b1, w2, b2, v


for i, (x, y) in zip(range(1000), mnist()):
    w1, b1, w2, b2, loss = optim(w1, b1, w2, b2, y, x)
    print(loss)
