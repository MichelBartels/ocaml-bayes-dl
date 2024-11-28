import jax.numpy as jnp
from jax import value_and_grad, jit
import numpy as np
from mnist import mnist
from tqdm import tqdm
from matplotlib import pyplot as plt
from optax import adamw, sgd, apply_updates

w1_encoder = np.random.normal(size=(784, 512)).astype(np.float32) * 0.01
b1_encoder = np.zeros((512,), dtype=np.float32)
w2_encoder = np.random.normal(size=(512, 16)).astype(np.float32) * 0.01
b2_encoder = np.zeros((16,), dtype=np.float32)
w3_encoder = np.random.normal(size=(512, 16)).astype(np.float32) * 0.01
b3_encoder = np.zeros((16,), dtype=np.float32)

w1_decoder = np.random.normal(size=(16, 512)).astype(np.float32) * 0.01
b1_decoder = np.zeros((512,), dtype=np.float32)
w2_decoder = np.random.normal(size=(512, 784)).astype(np.float32) * 0.01
b2_decoder = np.zeros((784,), dtype=np.float32)

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def relu(x):
    return jnp.maximum(0, x)

def tanh(x):
    return jnp.tanh(x)

def encoder(x, w1_encoder, b1_encoder, w2_encoder, b2_encoder, w3_encoder, b3_encoder):
    hidden = tanh(jnp.dot(x, w1_encoder) + b1_encoder)
    mean = jnp.dot(hidden, w2_encoder) + b2_encoder
    log_var = jnp.dot(hidden, w3_encoder) + b3_encoder
    return mean, log_var

def reparameterize(mean, log_var):
    std = jnp.exp(log_var)
    eps = np.random.normal(size=std.shape).astype(np.float32)
    return mean + std * eps

def decoder(z, w1_decoder, b1_decoder, w2_decoder, b2_decoder):
    hidden = tanh(jnp.dot(z, w1_decoder) + b1_decoder)
    mean = sigmoid(jnp.dot(hidden, w2_decoder) + b2_decoder)
    return mean

def mse(y_true, y_pred):
    return jnp.sum(jnp.square(y_true - y_pred), -1)

def kl(mean, log_var):
    return -0.5 * jnp.sum(1 + log_var - jnp.square(mean) - jnp.exp(log_var), -1)

def f(w1_encoder, b1_encoder, w2_encoder, b2_encoder, w3_encoder, b3_encoder, w1_decoder, b1_decoder, w2_decoder, b2_decoder, x):
    mean, log_var = encoder(x, w1_encoder, b1_encoder, w2_encoder, b2_encoder, w3_encoder, b3_encoder)
    z = reparameterize(mean, log_var)
    y_pred = decoder(z, w1_decoder, b1_decoder, w2_decoder, b2_decoder)
    loss = mse(x, y_pred) + kl(mean, log_var)
    return jnp.mean(loss)

lr = 0.02

params = [w1_encoder, b1_encoder, w2_encoder, b2_encoder, w3_encoder, b3_encoder, w1_decoder, b1_decoder, w2_decoder, b2_decoder]

opt = sgd(learning_rate=lr)
opt_state = opt.init(tuple(params))

params = [*params, opt_state]

@jit
def optim(params, x):
    [*params, opt_state] = params
    v, grad = value_and_grad(f, argnums=range(len(params)))(*params, x)
    updates, opt_state = opt.update(grad, opt_state, tuple(params))
    params = apply_updates(tuple(params), updates)
    return [*params, opt_state], v


num_steps = 25000

bar = tqdm(zip(range(num_steps), mnist()), total=num_steps)

for i, (x, _) in bar:
    params, loss = optim(params, x)
    bar.set_description(f"Loss: {loss}")
    if (i + 1) % 2500 == 0:
        y_pred = decoder(encoder(x[0], *params[:6])[0], *params[6:10])
        plt.imshow(x[0].reshape(28, 28), cmap='gray')
        plt.show()
        plt.imshow(y_pred.reshape(28, 28), cmap='gray')
        plt.show()
