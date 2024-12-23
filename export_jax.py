import jax.numpy as jnp
from jax import value_and_grad, jit, random, export, ShapeDtypeStruct
import numpy as np
from mnist import mnist, batch_size
from tqdm import tqdm
from matplotlib import pyplot as plt
from optax import adamw, sgd, apply_updates
from optax._src.transform import ScaleByAdamState
from optax._src.base import EmptyState

def init(shape):
    return (np.zeros(shape=shape).astype(np.float32),
     np.ones(shape=shape).astype(np.float32) * 0.001)

w1_encoder = init((784, 512))
b1_encoder = init((512,))
w2_encoder = init((512, 16))
b2_encoder = init((16,))
w3_encoder = init((512, 16))
b3_encoder = init((16,))

w1_decoder = init((16, 512))
b1_decoder = init((512,))
w2_decoder = init((512, 784))
b2_decoder = init((784,))

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def relu(x):
    return jnp.maximum(0, x)

def tanh(x):
    return jnp.tanh(x)

def reparameterize(seed, mean, std, add_batch=True):
    key_1, key_2 = random.split(seed)
    eps = random.normal(key_1, [batch_size, *mean.shape] if add_batch else mean.shape)
    return key_2, mean + std * eps

def encoder(seed, x, w1_encoder, b1_encoder, w2_encoder, b2_encoder, w3_encoder, b3_encoder):
    seed, w1_encoder = reparameterize(seed, *w1_encoder)
    seed, b1_encoder = reparameterize(seed, *b1_encoder)
    hidden = tanh(jnp.matmul(jnp.expand_dims(x, 1), w1_encoder) + jnp.expand_dims(b1_encoder, 1))
    seed, w2_encoder = reparameterize(seed, *w2_encoder)
    seed, b2_encoder = reparameterize(seed, *b2_encoder)
    mean = jnp.matmul(hidden, w2_encoder) + jnp.expand_dims(b2_encoder, 1)
    seed, w3_encoder = reparameterize(seed, *w3_encoder)
    seed, b3_encoder = reparameterize(seed, *b3_encoder)
    log_var = jnp.matmul(hidden, w3_encoder) + jnp.expand_dims(b3_encoder, 1)
    return seed, mean, log_var

def decoder(seed, z, w1_decoder, b1_decoder, w2_decoder, b2_decoder):
    seed, w1_decoder = reparameterize(seed, *w1_decoder)
    seed, b1_decoder = reparameterize(seed, *b1_decoder)
    hidden = tanh(jnp.matmul(z, w1_decoder) + jnp.expand_dims(b1_decoder, 1))
    seed, w2_decoder = reparameterize(seed, *w2_decoder)
    seed, b2_decoder = reparameterize(seed, *b2_decoder)
    mean = sigmoid(jnp.matmul(hidden, w2_decoder) + jnp.expand_dims(b2_decoder, 1))
    return seed, mean

def mse(y_true, y_pred):
    return jnp.sum(jnp.square(y_true - y_pred), -1)

def kl(mean, var, log_var):
    return -0.5 * jnp.sum(1 + log_var - jnp.square(mean) - var, -1)

def kl_param(param):
    mean, var = param
    var *= 1000
    return kl(mean, var, jnp.log(var))

def f(seed, w1_encoder, b1_encoder, w2_encoder, b2_encoder, w3_encoder, b3_encoder, w1_decoder, b1_decoder, w2_decoder, b2_decoder, x):
    seed, mean, log_var = encoder(seed, x, w1_encoder, b1_encoder, w2_encoder, b2_encoder, w3_encoder, b3_encoder)
    seed, z = reparameterize(seed, mean, jnp.exp(log_var), add_batch=False)
    seed, y_pred = decoder(seed, z, w1_decoder, b1_decoder, w2_decoder, b2_decoder)
    mse_ = jnp.mean(mse(jnp.expand_dims(x, 1), y_pred))
    kl_ = jnp.mean(kl(mean, jnp.exp(log_var), log_var))
    kl_params = jnp.mean(kl_param(w1_encoder)) + jnp.mean(kl_param(b1_encoder)) + jnp.mean(kl_param(w2_encoder)) + jnp.mean(kl_param(b2_encoder)) + jnp.mean(kl_param(w3_encoder)) + jnp.mean(kl_param(b3_encoder)) + jnp.mean(kl_param(w1_decoder)) + jnp.mean(kl_param(b1_decoder)) + jnp.mean(kl_param(w2_decoder)) + jnp.mean(kl_param(b2_decoder))
    loss = mse_ + kl_ + kl_params
    return loss

lr = 0.0001

params = [w1_encoder, b1_encoder, w2_encoder, b2_encoder, w3_encoder, b3_encoder, w1_decoder, b1_decoder, w2_decoder, b2_decoder]

opt = adamw(learning_rate=lr)
opt_state = opt.init(tuple(params))

params = [*params, opt_state]

@jit
def optim(seed, params, x):
    [*params, opt_state] = params
    v, grad = value_and_grad(f, argnums=range(1, len(params) + 1))(seed, *params, x)
    updates, opt_state = opt.update(grad, opt_state, tuple(params))
    params = apply_updates(tuple(params), updates)
    return [*params, opt_state], v


num_steps = 25000

bar = tqdm(zip(range(num_steps), mnist()), total=num_steps)

for i, (x, _) in bar:
    seed = random.key(i)
    if i == 0:
        #input_shapes = [ShapeDtypeStruct(seed.shape, seed.dtype)] + [ShapeDtypeStruct(x.shape, x.dtype) for x in params] + [ShapeDtypeStruct(x.shape, x.dtype)]
        export.register_namedtuple_serialization(ScaleByAdamState, serialized_name="ScaleByAdamState")
        export.register_namedtuple_serialization(EmptyState, serialized_name="EmptyState")
        stable_hlo = export.export(optim)(seed, params, x).mlir_module_serialized
        with open("train.mlir", "wb") as f:
            f.write(stable_hlo)
    params, loss = optim(seed, params, x)
    bar.set_description(f"Loss: {loss}")
    if (i + 1) % 2500 == 0:
        seed, mean, _ = encoder(seed, x, *params[:6])
        _, y_pred = decoder(seed, mean, *params[6:10])
        plt.imshow(x[0].reshape(28, 28), cmap='gray')
        plt.show()
        plt.imshow(y_pred[0].reshape(28, 28), cmap='gray')
        plt.show()
