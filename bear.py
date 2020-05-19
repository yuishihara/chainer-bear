import numpy as np

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
        raise NotImplementedError('Use multiplication operator * directly')

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


class BEAR(object):
    def __init__(self, critic_builder, actor_builder, vae_builder, state_dim, action_dim, *,
                 gamma=0.99, tau=0.5*1e-3, lmb=0.75, epsilon=0.05, stddev_coeff=0.4,
                 num_q_ensembles=2, num_mmd_actions=5, batch_size=100, device=-1):
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

        if 0 < device:
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
        self._num_mmd_actions = num_mmd_actions
        delta_conf = 0.1
        self._stddev_coeff = stddev_coeff * \
            np.sqrt((1 - delta_conf) / delta_conf)

        self._batch_size = batch_size
        self._device = device

        self._initialized = False

        self._num_iterations = 0

    def train(self, iterator, **kwargs):
        if not self._initialized:
            self._initialize_all_networks()
            self._initialized = True

        batch = concat_examples(iterator.next(), device=self._device)

        self._q_update(batch)
        self._policy_update(batch)
        self._update_all_target_networks(tau=self._tau)

        self._num_iterations += 1

    def _q_update(self, batch):
        (s, a, _, _, _) = batch
        target_q_value = self._compoute_target_q_value(batch)

        for q, optimizer in zip(self._q_ensembles, self._q_optimizers):
            loss = F.mean_squared_error(target_q_value, q(s, a))
            optimizer.target.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    def _compoute_target_q_value(self, batch, *, num_action_samples=10):
        (_, _, r, s_next, non_terminal) = batch

        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
            s_next_rep = F.repeat(
                x=s_next, repeats=num_action_samples, axis=0)
            a_next_rep = self._target_pi.sample(s_next_rep)

            q_values = F.stack([q_target(s_next_rep, a_next_rep)
                                for q_target in self._target_q_ensembles])
            weighted_q_minmax = self._lambda * F.min(q_values, axis=0, keepdims=True) \
                + (1 - self._lambda) * F.max(q_values, axis=0, keepdims=True)
            next_q_value = F.max(
                F.reshape(weighted_q_minmax, shape=(self._batch_size, -1)), keepdims=True)
            target_q_value = r + self._gamma * next_q_value * non_terminal
            target_q_value.unchain()
            assert target_q_value.shape == (self._batch_size, 1)
        return target_q_value

    def _policy_update(self, batch):
        self._train_vae(batch)

        _, raw_sampled_actions = self._vae._decode_multiple(
            s, decode_num=self._num_mmd_actions)
        pi_actions, raw_pi_actions = self._pi._sample_multiple(
            s, sample_num=self._num_mmd_actions)

        mmd_loss = self._compute_gaussian_mmd(
            raw_sampled_actions, raw_pi_actions)

        s_hat = F.expand_dims(s, axis=0)
        s_hat = F.repeat(s_hat, repeats=self._num_mmd_actions, axis=0)
        s_hat = F.reshape(s_hat, shape(self._batch_size *
                                       self._num_mmd_actions, s.shape[-1]))
        a_hat = F.transpose(pi_actions, axes=(1, 0, 2))
        a_hat = F.reshape(a_hat, shape(self._batch_size *
                                       self._num_mmd_actions, a.shape[-1]))

        q_values = F.stack([q(s_hat, a_hat) for q in self._q_ensembles])
        q_values = F.reshape(q_values, shape=(
            self._num_q_ensembles, self._num_mmd_actions * self._batch_size, 1))
        q_values = F.mean(q_values, axis=1)
        q_stddev = self._compute_stddev(x=q_values, axis=0, keepdims=False)

        q_min = F.min(q_values, axis=0)

        assert q_min.shape == q_stddev.shape

        if self._num_iterations > 10000:
            pi_loss = F.mean(-q_min +
                             q_stddev * self._stddev_coeff +
                             self._lagrange_multiplier.exp() * mmd_loss)
        else:
            pi_loss = F.mean(self._lagrange_multiplier.exp() * mmd_loss)

        # Dual gradient descent
        # Update actor
        self._pi_optimizer.target.cleargrads()
        pi_loss.backward()
        self._pi_optimizer.update()

        # Update lagrange multiplier
        lagrange_loss = -F.mean(-q_min +
                                q_stddev * self._stddev_coeff +
                                self._lagrange_multiplier.exp() * (mmd_loss - self._epsilon))
        self._lagrange_optimizer.target.cleargrads()
        lagrange_loss.backward()
        self._lagrange_optimizer.update()

        pi_loss.unchain_backward()
        lagrange_loss.unchain_backward()

        # Clip lagrange multiplier in range
        self._lagrange_multiplier.clip(-5.0, 10.0)

    def _train_vae(self, batch):
        (s, a, _, _, _) = batch
        reconstructed_action, mean, ln_var = self._vae((s, a))
        reconstruction_loss = F.mean_squared_error(reconstructed_action, a)
        latent_loss = 0.5 * \
            F.gaussian_kl_divergence(mean, ln_var, reduce='mean')
        vae_loss = reconstruction_loss + latent_loss

        self._vae_optimizer.target.cleargrads()
        vae_loss.backward()
        vae_loss.unchain_backward()
        self._vae.optimizer.update()

    def _compute_stddev(self, x, axis=None, keepdims=False):
        # stddev = sqrt(E[X^2] - E[X]^2)
        return F.sqrt(F.mean(x**2, axis=axis, keepdims=keepdims) - F.mean(x, axis=axis, keepdims=keepdims)**2)

    def _initialize_all_networks(self):
        self._update_all_target_networks(tau=1.0)

    def _update_all_target_networks(self, tau):
        for target_q, q in zip(self._target_q_ensembles, self._q_ensembles):
            self._update_target_network(target_q, _q, tau)
        self._update_target_network(self._target_pi, self._pi, tau)

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.params(), origin.params()):
            target_param.data = tau * origin_param.data + \
                (1.0 - tau) * target_param.data

    @staticmethod
    def _compute_gaussian_mmd(samples1, samples2, *, sigma=0.2):
        n = samples1.shape[1]
        m = samples2.shape[1]

        k_xx = F.expand_dims(x=samples1, axis=2) - \
            F.expand_dims(x=samples1, axis=1)
        sum_k_xx = F.sum(F.exp(-F.sum(k_xx**2, axis=-1) / (2.0*sigma)))

        k_xy = F.expand_dims(x=samples1, axis=2) - \
            F.expand_dims(x=samples2, axis=1)
        sum_k_xy = F.sum(F.exp(-F.sum(k_xy**2, axis=-1) / (2.0*sigma)))

        k_yy = F.expand_dims(x=samples2, axis=2) - \
            F.expand_dims(x=samples2, axis=1)
        sum_k_yy = F.sum(F.exp(-F.sum(k_yy**2, axis=-1) / (2.0*sigma)))

        mmd_squared = sum_k_xx / (n*n) - 2.0 * sum_k_xy / \
            (m*n) + sum_k_yy / (m*m)

        return F.sqrt(mmd_squared + 1e-6)


if __name__ == "__main__":
    def _compute_sum_gaussian_kernel(s1, s2):
        kernel_sum = 0.0
        for i in range(s1.shape[1]):
            for j in range(s2.shape[1]):
                diff = 0.0
                for k in range(s1.shape[2]):
                    diff += (s1[0][i][k] - s2[0][j][k]) ** 2
                kernel_sum += np.exp(-diff / 2.0)
        return kernel_sum

    samples1 = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]], dtype=np.float32)
    samples2 = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)

    n = samples1.shape[1]
    m = samples2.shape[1]

    expected_kxx = _compute_sum_gaussian_kernel(samples1, samples1)
    expected_kxy = _compute_sum_gaussian_kernel(samples1, samples2)
    expected_kyy = _compute_sum_gaussian_kernel(samples2, samples2)

    expected_mmd = expected_kxx/(n*n) - 2.0*expected_kxy / \
        (m*n) + expected_kyy/(m*m)
    expected_mmd = np.sqrt(expected_mmd + 1e-6)
    actual_mmd = BEAR._compute_gaussian_mmd(samples1, samples2, sigma=1.0)
    actual_mmd = actual_mmd.array
    print('actual mmd: ', actual_mmd)
    print('expected mmd: ', expected_mmd)
    assert np.isclose(actual_mmd, expected_mmd)

    lagrange_multiplier = OptimizableLagrangeMultiplier(initial_value=3.0)
    scaled_multiplier = lagrange_multiplier * 5
    print(
        f'lagrange multiplier: {lagrange_multiplier}, scaled multiplier {scaled_multiplier}')
