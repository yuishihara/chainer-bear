import chainer
import chainer.functions as F
import chainer.links as L

from .torch_uniform_init import HeUniformTorch, LinearBiasInitializerTorch


class _Actor(chainer.Chain):
    def save(self, path):
        if path.exists():
            raise ValueError('File already exist')
        chainer.serializers.save_npz(path.resolve(), self)

    def load(self, path):
        if not path.exists():
            raise ValueError('File {} not found'.format(path))
        chainer.serializers.load_npz(path.resolve(), self)


class MujocoActor(_Actor):
    def __init__(self, state_dim, action_dim):
        super(MujocoActor, self).__init__()
        initialW = HeUniformTorch()
        with self.init_scope():
            self._linear1 = L.Linear(
                in_size=state_dim, out_size=400, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=state_dim))
            self._linear2 = L.Linear(
                in_size=400, out_size=300, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=400))

            self._linear_mean = L.Linear(
                in_size=300, out_size=action_dim, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=300))
            self._linear_ln_var = L.Linear(
                in_size=300, out_size=action_dim, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=300))

        self._action_dim = action_dim

    def __call__(self, s):
        h = self._linear1(s)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)

        mean = self._linear_mean(h)
        return F.tanh(mean)

    def _sample(self, s):
        h = self._linear1(s)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)

        mean = self._linear_mean(h)
        ln_var = self._linear_ln_var(h)
        z = F.gaussian(mean, ln_var)
        return F.tanh(z), z

    def _sample_multiple(self, s, sample_num):
        h = self._linear1(s)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)

        mean = self._linear_mean(h)
        ln_var = self._linear_ln_var(h)

        batch_size = s.shape[0]
        stddev = F.exp(ln_var * mean.dtype.type(0.5))
        xp = chainer.backend.get_array_module(s)
        noise = chainer.Variable(xp.random.normal(loc=0,
                                                  scale=1,
                                                  size=(batch_size, sample_num, self._action_dim)))
        noise = F.cast(noise, typ=xp.float32)

        z = F.expand_dims(mean, axis=1) + \
            F.expand_dims(stddev, axis=1) * F.clip(noise, -0.5, 0.5)

        return F.tanh(z), z


class VAEActor(_Actor):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(VAEActor, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._latent_dim = latent_dim
        initialW = HeUniformTorch()
        with self.init_scope():
            self._linear1 = L.Linear(
                in_size=(state_dim + action_dim), out_size=750, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=state_dim))
            self._linear2 = L.Linear(
                in_size=750, out_size=750, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=750))

            self._linear_mean = L.Linear(
                in_size=750, out_size=latent_dim, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=750))
            self._linear_ln_var = L.Linear(
                in_size=750, out_size=latent_dim, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=750))

            self._linear3 = L.Linear(
                in_size=(state_dim + latent_dim), out_size=750, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=state_dim + latent_dim))
            self._linear4 = L.Linear(
                in_size=750, out_size=750, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=750))
            self._linear5 = L.Linear(
                in_size=750, out_size=action_dim, initialW=initialW, initial_bias=LinearBiasInitializerTorch(fan_in=750))

    def __call__(self, x):
        (s, a) = x
        z, mu, ln_var = self._encode(s, a)
        reconstructed, _ = self._decode(s, z)
        return reconstructed, mu, ln_var

    def _encode(self, s, a):
        mu, ln_var = self._latent_distribution(s, a)
        # 2 * ln_std = ln_var
        # original code is written in ln_std form
        # Clip for numerical stability
        ln_var = F.clip(ln_var, x_min=-8, x_max=30)
        return F.gaussian(mu, ln_var), mu, ln_var

    def _decode(self, s, z=None):
        if z is None:
            xp = chainer.backend.get_array_module(s)
            z = chainer.Variable(xp.random.normal(
                0, 1, size=(s.shape[0], self._latent_dim)))
            z = F.cast(z, typ=xp.float32)
            z = F.clip(z, -0.5, 0.5)
        x = F.concat((s, z), axis=1)
        h = self._linear3(x)
        h = F.relu(h)
        h = self._linear4(h)
        h = F.relu(h)
        h = self._linear5(h)

        return F.tanh(h), h

    def _decode_multiple(self, s, z=None, decode_num=10):
        if z is None:
            xp = chainer.backend.get_array_module(s)
            z = chainer.Variable(xp.random.normal(
                0, 1, size=(s.shape[0], decode_num, self._latent_dim)))
            z = F.cast(z, typ=xp.float32)
            z = F.clip(z, -0.5, 0.5)

        s = F.expand_dims(s, axis=0)
        s = F.repeat(s, repeats=decode_num, axis=0)
        s = F.transpose(s, axes=(1, 0, 2))

        x = F.concat((s, z), axis=2)
        x = F.reshape(x, shape=(-1, x.shape[-1]))
        h = self._linear3(x)
        h = F.relu(h)
        h = self._linear4(h)
        h = F.relu(h)
        h = self._linear5(h)
        h = F.reshape(h, shape=(-1, decode_num, h.shape[-1]))

        return F.tanh(h), h

    def _latent_distribution(self, s, a):
        x = F.concat((s, a), axis=1)
        h = self._linear1(x)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)

        mu = self._linear_mean(h)
        ln_var = self._linear_ln_var(h)

        return mu, ln_var
