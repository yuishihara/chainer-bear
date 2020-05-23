import numpy as np

import pathlib

import logging

import chainer
import chainer.functions as F
from chainer import optimizers
from chainer.dataset import concat_examples


class OptimizableLagrangeMultiplier(chainer.Chain):
    def __init__(self, initial_value=None):
        super(OptimizableLagrangeMultiplier, self).__init__()

        with self.init_scope():
            initializer = initial_value if initial_value is not None else chainer.initializers.Normal(
                scale=1.0)
            self._lagrange_multiplier = chainer.Parameter(
                initializer=initializer, shape=(1)
            )
        self.clip(-5.0, 10.0)

    def __call__(self):
        return self._lagrange_multiplier

    def __mul__(self, other):
        return self._lagrange_multiplier * other

    def __str__(self):
        return str(self._lagrange_multiplier)

    def exp(self):
        return F.exp(self._lagrange_multiplier)

    def clip(self, min_value, max_value):
        xp = chainer.backend.get_array_module(self)
        self._lagrange_multiplier.data = xp.clip(
            self._lagrange_multiplier.data, min_value, max_value)

    def save(self, path):
        if path.exists():
            raise ValueError('File already exist')
        chainer.serializers.save_npz(path.resolve(), self)

    def load(self, path):
        if not path.exists():
            raise ValueError('File {} not found'.format(path))
        chainer.serializers.load_npz(path.resolve(), self)


class BEAR(object):
    def __init__(self, critic_builder, actor_builder, vae_builder, state_dim, action_dim, *,
                 gamma=0.99, tau=0.5 * 1e-3, lmb=0.75, epsilon=0.05, stddev_coeff=0.4, mmd_sigma=20.0,
                 warmup_iterations=100000, num_q_ensembles=2, num_mmd_samples=5, batch_size=100,
                 kernel_type='laplacian', device=-1):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._q_ensembles = []
        self._target_q_ensembles = []
        self._q_optimizers = []

        for _ in range(num_q_ensembles):
            q_function = critic_builder(state_dim, action_dim)
            target_q_function = critic_builder(state_dim, action_dim)

            q_optimizer = optimizers.Adam()
            q_optimizer.setup(q_function)

            self._q_ensembles.append(q_function)
            self._target_q_ensembles.append(target_q_function)
            self._q_optimizers.append(q_optimizer)

        self._pi = actor_builder(state_dim, action_dim)
        self._target_pi = actor_builder(state_dim, action_dim)
        self._pi_optimizer = optimizers.Adam()
        self._pi_optimizer.setup(self._pi)

        self._vae = vae_builder(state_dim, action_dim)
        self._vae_optimizer = optimizers.Adam()
        self._vae_optimizer.setup(self._vae)

        self._lagrange_multiplier = OptimizableLagrangeMultiplier()
        self._lagrange_optimizer = optimizers.Adam(alpha=1e-3)
        self._lagrange_optimizer.setup(self._lagrange_multiplier)

        if not device < 0:
            for q_function in self._q_ensembles:
                q_function.to_device(device=device)

            for target_q_function in self._target_q_ensembles:
                target_q_function.to_device(device=device)

            self._pi.to_device(device=device)
            self._target_pi.to_device(device=device)
            self._vae.to_device(device=device)
            self._lagrange_multiplier.to_device(device=device)

        self._gamma = 0.99
        self._tau = tau
        self._lambda = lmb
        self._epsilon = epsilon
        self._num_q_ensembles = num_q_ensembles
        self._num_mmd_samples = num_mmd_samples
        self._mmd_sigma = mmd_sigma
        delta_conf = 0.1
        self._stddev_coeff = stddev_coeff * \
            np.sqrt((1 - delta_conf) / delta_conf)
        self._kernel_type = kernel_type

        self._batch_size = batch_size
        self._device = device

        self._initialized = False

        self._num_iterations = 0
        self._warmup_iterations = warmup_iterations

    def train(self, iterator, **kwargs):
        if not self._initialized:
            self._initialize_all_networks()
            self._initialized = True

        batch = concat_examples(iterator.next(), device=self._device)
        status = {}
        q_update_status = self._q_update(batch)
        policy_update_status = self._policy_update(batch)
        self._update_all_target_networks(tau=self._tau)

        self._num_iterations += 1

        status.update(q_update_status)
        status.update(policy_update_status)

        return status

    def compute_action(self, s):
        with chainer.using_config('enable_backprop', False), chainer.using_config('train', False):
            s = np.float32(s)
            if s.ndim == 1:
                s = np.reshape(s, newshape=(1, ) + s.shape)
            state = chainer.Variable(s)
            if not self._device < 0:
                state.to_gpu()
            a = self._pi(state)
            if not self._device < 0:
                a.to_cpu()

            if a.shape[0] == 1:
                return np.squeeze(a.data, axis=0)
            else:
                return a.data

    def save_models(self, outdir, prefix):
        for index, q_func in enumerate(self._q_ensembles):
            q_filepath = pathlib.Path(
                outdir, 'q{}_iter-{}'.format(index, prefix))
            q_func.to_cpu()
            q_func.save(q_filepath)
            if not self._device < 0:
                q_func.to_device(device=self._device)

        pi_filepath = pathlib.Path(outdir, 'pi_iter-{}'.format(prefix))
        vae_filepath = pathlib.Path(outdir, 'vae_iter-{}'.format(prefix))
        lagrange_filepath = pathlib.Path(
            outdir, 'lagrange_iter-{}'.format(prefix))

        self._pi.to_cpu()
        self._vae.to_cpu()
        self._lagrange_multiplier.to_cpu()

        self._pi.save(pi_filepath)
        self._vae.save(vae_filepath)
        self._lagrange_multiplier.save(lagrange_filepath)

        if not self._device < 0:
            self._pi.to_device(device=self._device)
            self._vae.to_device(device=self._device)
            self._lagrange_multiplier.to_device(device=self._device)

    def load_models(self, q_param_filepaths, pi_filepath, vae_filepath, lagrange_filepath):
        for index, q_func in enumerate(self._q_ensembles):
            q_func.to_cpu()
            if q_param_filepaths:
                q_func.load(q_param_filepaths[index])
            if not self._device < 0:
                q_func.to_device(device=self._device)

        self._pi.to_cpu()
        self._vae.to_cpu()
        self._lagrange_multiplier.to_cpu()

        if pi_filepath:
            self._pi.load(pi_filepath)
        if vae_filepath:
            self._vae.load(vae_filepath)
        if lagrange_filepath:
            self._lagrange_multiplier.load(lagrange_filepath)

        if not self._device < 0:
            self._pi.to_device(device=self._device)
            self._vae.to_device(device=self._device)
            self._lagrange_multiplier.to_device(device=self._device)

    def _q_update(self, batch):
        status = {}

        (s, a, _, _, _) = batch
        target_q_value = self._compute_target_q_value(batch)

        for optimizer in self._q_optimizers:
            optimizer.target.cleargrads()

        loss = 0.0
        for q in self._q_ensembles:
            loss += F.mean_squared_error(target_q_value, q(s, a))

        loss.backward()
        loss.unchain_backward()

        for optimizer in self._q_optimizers:
            optimizer.update()

        xp = chainer.backend.get_array_module(loss)
        status['q_loss'] = xp.array(loss.array)
        return status

    def _compute_target_q_value(self, batch, *, num_action_samples=10):
        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
            (_, _, r, s_next, non_terminal) = batch
            r = F.reshape(r, shape=(*r.shape, 1))
            non_terminal = F.reshape(
                non_terminal, shape=(*non_terminal.shape, 1))

            s_next_rep = F.repeat(x=s_next, repeats=num_action_samples, axis=0)
            a_next_rep, _ = self._target_pi._sample(s_next_rep)
            q_values = F.stack([q_target(s_next_rep, a_next_rep)
                                for q_target in self._target_q_ensembles])
            assert q_values.shape == (
                self._num_q_ensembles, self._batch_size * num_action_samples, 1)

            weighted_q_minmax = self._lambda * F.min(q_values, axis=0) \
                + (1 - self._lambda) * F.max(q_values, axis=0)
            assert weighted_q_minmax.shape == (
                self._batch_size * num_action_samples, 1)
            next_q_value = F.max(
                F.reshape(weighted_q_minmax, shape=(self._batch_size, -1)), axis=1, keepdims=True)
            assert next_q_value.shape == (self._batch_size, 1)
            target_q_value = r + self._gamma * next_q_value * non_terminal
            target_q_value.unchain()
            assert target_q_value.shape == (self._batch_size, 1)
        return target_q_value

    def _policy_update(self, batch):
        status = {}
        vae_status = self._train_vae(batch)

        status.update(vae_status)

        (s, a, _, _, _) = batch
        _, raw_sampled_actions = self._vae._decode_multiple(
            s, decode_num=self._num_mmd_samples)
        pi_actions, raw_pi_actions = self._pi._sample_multiple(
            s, sample_num=self._num_mmd_samples)

        if self._kernel_type == 'gaussian':
            mmd_loss = self._compute_gaussian_mmd(
                raw_sampled_actions, raw_pi_actions, sigma=self._mmd_sigma)
        elif self._kernel_type == 'laplacian':
            mmd_loss = self._compute_laplacian_mmd(
                raw_sampled_actions, raw_pi_actions, sigma=self._mmd_sigma)
            )
        else:
            raise ValueError('Unknown kernel: {}'.format(self._kernel_type))
        assert mmd_loss.shape == (self._batch_size, 1)

        s_hat = F.expand_dims(s, axis=0)
        s_hat = F.repeat(s_hat, repeats=self._num_mmd_samples, axis=0)
        s_hat = F.reshape(s_hat, shape=(self._batch_size * self._num_mmd_samples,
                                        s.shape[-1]))
        a_hat = F.transpose(pi_actions, axes=(1, 0, 2))
        a_hat = F.reshape(a_hat, shape=(self._batch_size * self._num_mmd_samples,
                                        a.shape[-1]))

        q_values = F.stack([q(s_hat, a_hat) for q in self._q_ensembles])
        assert q_values.shape == (
            self._num_q_ensembles, self._batch_size * self._num_mmd_samples, 1)
        q_values = F.reshape(q_values, shape=(
            self._num_q_ensembles, self._num_mmd_samples, self._batch_size,  1))
        q_values = F.mean(q_values, axis=1)
        assert q_values.shape == (self._num_q_ensembles, self._batch_size, 1)
        q_stddev = self._compute_stddev(x=q_values, axis=0, keepdims=False)

        q_min = F.min(q_values, axis=0)

        assert q_min.shape == q_stddev.shape
        assert q_min.shape == (self._batch_size, 1)

        if self._num_iterations > self._warmup_iterations:
            pi_loss = F.mean(-q_min
                             + q_stddev * self._stddev_coeff
                             + self._lagrange_multiplier.exp() * mmd_loss)
        else:
            pi_loss = F.mean(self._lagrange_multiplier.exp() * mmd_loss)

        # Dual gradient descent
        # Update actor
        self._pi_optimizer.target.cleargrads()
        pi_loss.backward()
        self._pi_optimizer.update()

        # Update lagrange multiplier
        lagrange_loss = -F.mean(-q_min
                                + q_stddev * self._stddev_coeff
                                + self._lagrange_multiplier.exp() * (mmd_loss - self._epsilon))
        self._lagrange_optimizer.target.cleargrads()
        lagrange_loss.backward()
        self._lagrange_optimizer.update()

        pi_loss.unchain_backward()
        lagrange_loss.unchain_backward()

        # Clip lagrange multiplier in range
        self._lagrange_multiplier.clip(-5.0, 10.0)

        xp = chainer.backend.get_array_module(pi_loss)
        status['pi_loss'] = xp.array(pi_loss.array)
        status['mmd_loss'] = xp.mean(xp.array(mmd_loss.array))
        status['lagrange_loss'] = xp.array(pi_loss.array)
        status['lagrange_multiplier'] = xp.array(
            self._lagrange_multiplier().array)

        return status

    def _train_vae(self, batch):
        status = {}

        (s, a, _, _, _) = batch
        reconstructed_action, mean, ln_var = self._vae((s, a))
        reconstruction_loss = F.mean_squared_error(reconstructed_action, a)
        latent_loss = 0.5 * \
            F.gaussian_kl_divergence(mean, ln_var, reduce='mean')
        vae_loss = reconstruction_loss + latent_loss

        self._vae_optimizer.target.cleargrads()
        vae_loss.backward()
        vae_loss.unchain_backward()
        self._vae_optimizer.update()

        xp = chainer.backend.get_array_module(vae_loss)
        status['vae_loss'] = xp.array(vae_loss.array)
        return status

    def _compute_stddev(self, x, axis=None, keepdims=False):
        # stddev = sqrt(E[(X-E[X])^2])
        mu = F.mean(x, axis=axis, keepdims=keepdims)
        mu.unchain()
        var = F.mean((x - mu) ** 2, axis=axis, keepdims=keepdims)
        return F.sqrt(var)

    def _initialize_all_networks(self):
        self._update_all_target_networks(tau=1.0)

    def _update_all_target_networks(self, tau):
        for target_q, q in zip(self._target_q_ensembles, self._q_ensembles):
            self._update_target_network(target_q, q, tau)
        self._update_target_network(self._target_pi, self._pi, tau)

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.params(), origin.params()):
            target_param.data = tau * origin_param.data + \
                (1.0 - tau) * target_param.data

    def _compute_gaussian_mmd(self, samples1, samples2, *, sigma=20.0):
        n = samples1.shape[1]
        m = samples2.shape[1]

        k_xx = F.expand_dims(x=samples1, axis=2) - \
            F.expand_dims(x=samples1, axis=1)
        sum_k_xx = F.sum(
            F.exp(-F.sum(k_xx**2, axis=-1, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        k_xy = F.expand_dims(x=samples1, axis=2) - \
            F.expand_dims(x=samples2, axis=1)
        sum_k_xy = F.sum(
            F.exp(-F.sum(k_xy**2, axis=-1, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        k_yy = F.expand_dims(x=samples2, axis=2) - \
            F.expand_dims(x=samples2, axis=1)
        sum_k_yy = F.sum(
            F.exp(-F.sum(k_yy**2, axis=-1, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        mmd_squared = \
            sum_k_xx / (n * n) - 2.0 * sum_k_xy / (m * n) + sum_k_yy / (m * m)

        return F.sqrt(mmd_squared + 1e-6)

    def _compute_laplacian_mmd(self, samples1, samples2, *, sigma=20.0):
        n = samples1.shape[1]
        m = samples2.shape[1]

        k_xx = F.expand_dims(x=samples1, axis=2) - \
            F.expand_dims(x=samples1, axis=1)
        sum_k_xx = F.sum(
            F.exp(-F.sum(F.absolute(k_xx), axis=-1, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        k_xy = F.expand_dims(x=samples1, axis=2) - \
            F.expand_dims(x=samples2, axis=1)
        sum_k_xy = F.sum(
            F.exp(-F.sum(F.absolute(k_xy), axis=-1, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        k_yy = F.expand_dims(x=samples2, axis=2) - \
            F.expand_dims(x=samples2, axis=1)
        sum_k_yy = F.sum(
            F.exp(-F.sum(F.absolute(k_yy), axis=-1, keepdims=True) / (2.0 * sigma)), axis=(1, 2))

        mmd_squared = \
            sum_k_xx / (n * n) - 2.0 * sum_k_xy / (m * n) + sum_k_yy / (m * m)

        return F.sqrt(mmd_squared + 1e-6)


if __name__ == "__main__":
    def _compute_sum_gaussian_kernel(s1, s2):
        gaussian_kernel = []
        for b in range(s1.shape[0]):
            kernel_sum = 0.0
            for i in range(s1.shape[1]):
                for j in range(s2.shape[1]):
                    diff = 0.0
                    for k in range(s1.shape[2]):
                        diff += (s1[b][i][k] - s2[b][j][k]) ** 2
                    kernel_sum += np.exp(-diff / (2.0 * 20.0))
            gaussian_kernel.append([kernel_sum])
        return np.array(gaussian_kernel)

    samples1 = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                         [[4, 4, 4], [2, 2, 2], [3, 3, 3]]], dtype=np.float32)
    samples2 = np.array([[[0, 0, 0], [1, 1, 1]],
                         [[1, 2, 3], [1, 1, 1]]], dtype=np.float32)

    n = samples1.shape[1]
    m = samples2.shape[1]

    expected_kxx = _compute_sum_gaussian_kernel(samples1, samples1)
    expected_kxy = _compute_sum_gaussian_kernel(samples1, samples2)
    expected_kyy = _compute_sum_gaussian_kernel(samples2, samples2)

    expected_mmd = \
        expected_kxx / (n * n) - 2.0 * expected_kxy / \
        (m * n) + expected_kyy / (m * m)
    expected_mmd = np.sqrt(expected_mmd + 1e-6)

    def fake_builder(self, *args):
        return chainer.Chain()

    bear = BEAR(critic_builder=fake_builder, actor_builder=fake_builder,
                vae_builder=fake_builder, state_dim=5, action_dim=5)
    actual_mmd = bear._compute_gaussian_mmd(samples1, samples2, sigma=20.0)
    actual_mmd = actual_mmd.array
    print('actual mmd: ', actual_mmd)
    print('expected mmd: ', expected_mmd)
    assert actual_mmd.shape == (samples1.shape[0], 1)
    assert np.all(np.isclose(actual_mmd, expected_mmd))

    lagrange_multiplier = OptimizableLagrangeMultiplier(initial_value=3.0)
    scaled_multiplier = lagrange_multiplier * 5
    print(
        f'lagrange multiplier: {lagrange_multiplier}, scaled multiplier {scaled_multiplier}')

    array = np.random.normal(loc=0.0, scale=1.0, size=(100, 100, 1))
    stddev = bear._compute_stddev(array, axis=0, keepdims=False)
    # print('stddev', stddev)
    assert stddev.shape == (100, 1)

    batch_size = 3
    num_samples = 10
    s = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]])
    s_hat = F.expand_dims(s, axis=0)
    s_hat = F.repeat(s_hat, repeats=num_samples, axis=0)
    assert s_hat.shape == (num_samples, batch_size,  5)
    s_reshaped = F.reshape(s_hat, shape=(
        batch_size * num_samples, s.shape[-1]))
    s_reshaped_back = F.reshape(s_reshaped, shape=(
        num_samples, batch_size, s.shape[-1]))
    assert np.all(s_hat.array == s_reshaped_back.array)

    array = np.array([[[1], [1], [1]], [[2], [2], [2]], [[3], [3], [3]]])
    print('array shape: ', array.shape)
    print('mean axis=0:', np.mean(array, axis=0))
    print('mean axis=1:', np.mean(array, axis=1))
