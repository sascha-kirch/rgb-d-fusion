import math
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf


class BetaSchedule:
    """Abstraction of the beta schedule modulating the timesteps in the diffusion process."""

    # Inspired by:
    # https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L18-L62

    def _betas_for_alpha_bar(
        self,
        num_diffusion_timesteps: int,
        alpha_bar: Callable[[float], tf.Tensor],
        max_beta: float = 0.999,
    ) -> tf.Tensor:
        """Create a beta schedule from an alpha_bar schedule.

        Beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta)
        over time from t = [0,1].

        Args:
            num_diffusion_timesteps (int): Number of betas to produce

            alpha_bar (Callable[[float], tf.Tensor]): Function that takes an argument t from 0 to 1 and produces the
                cumulative product of (1-beta) up to that part of the diffusion process.

            max_beta (float, optional): he maximum beta to use; use values lower than 1 to prevent singularities.
                Defaults to 0.999.

        Returns:
            tf.Tensor: Beta schedule.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return tf.constant(np.array(betas), dtype=self.policy.variable_dtype)

    def _get_beta_schedule(
        self,
        schedule: str,
        timesteps: int,
        start: float,
        stop: float,
    ) -> tf.Tensor:
        """Returns a beta schedule.

        Args:
            schedule (str): ["cosine" | "linear" | "sigmoid"]
            timesteps (int): Number of timesteps the schedule shall have
            start (float): Start value of the schedule
            stop (float): Stop value of the schedule

        Raises:
            ValueError: if schedule is not a valid option.

        Returns:
            tf.Tensor: Beta schedule.
        """
        if schedule == "linear":
            betas = tf.linspace(start, stop, timesteps)
        elif schedule == "sigmoid":
            betas = tf.linspace(-6, 6, timesteps)
            betas = tf.math.sigmoid(betas) * (stop - start) + start
        elif schedule == "cosine":
            s = 0.008
            betas = self._betas_for_alpha_bar(
                timesteps,
                lambda t: tf.math.cos((t + s) / (1 + s) * math.pi / 2) ** 2,
            )
        else:
            raise ValueError(f"Schedule {schedule} is not defined.")

        return tf.cast(betas, dtype=self.policy.variable_dtype)

    def __init__(
        self,
        schedule: str = "linear",
        timesteps: int = 1000,
        start: float = 1e-4,
        stop: float = 2e-2,
        k: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        """Abstraction of the beta schedule modulating the timesteps in the diffusion process.

        Args:
            schedule (str, optional): ["cosine" | "linear" | "sigmoid"]. Defaults to "linear".
            timesteps (int, optional): Number of timesteps the schedule shall have. Defaults to 1000.
            start (float, optional): Start value of the schedule. Defaults to 1e-4.
            stop (float, optional): Stop value of the schedule. Defaults to 2e-2.
            k (float, optional): k term for P2 loss scaling. Defaults to 1.0.
            gamma (float, optional): gamma term for P2 loss scaling. Defaults to 1.0.
        """
        self.config = {"schedule": schedule, "timesteps": timesteps, "start": start, "stop": stop}
        self.policy = tf.keras.mixed_precision.global_policy()
        self.betas = self._get_beta_schedule(schedule, timesteps, start, stop)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = tf.math.cumprod(self.alphas, 0)
        # shifts all elements by 1 to the right and adds 1 at the first position. last element is dropped due to [:-1]
        self.alpha_bar_prev = tf.concat(
            (tf.constant(np.array([1.0]), dtype=self.policy.variable_dtype), self.alpha_bar[:-1]), axis=0
        )
        # shifts all elements by 1 to the right and adds 1 at the first position. last element is dropped due to [:-1]
        self.alpha_bar_next = tf.concat(
            (self.alpha_bar[1:], tf.constant(np.array([0.0]), dtype=self.policy.variable_dtype)), axis=0
        )

        # Pre-calculated alpha coefficients
        self.one_minus_alpha_bar = 1.0 - self.alpha_bar
        self.sqrt_alpha_bar = tf.math.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = tf.math.sqrt(self.one_minus_alpha_bar)
        self.log_one_minus_alpha_bar = tf.math.log(self.one_minus_alpha_bar)
        self.sqrt_recip_alpha_bar = tf.math.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recip_alpha_bar_minus_one = tf.math.sqrt(1.0 / self.alpha_bar - 1)

        # Posterior Calculations q(x_{t-1} | xt, x0)
        self.posterior_var = self.betas * (1.0 - self.alpha_bar_prev) / self.one_minus_alpha_bar

        # [1:2] returs the element 1 in form of an tensor keeping the dimensions
        self.posterior_log_var_clipped = tf.math.log(
            tf.concat((self.posterior_var[1:2], self.posterior_var[1:]), axis=0)
        )
        # Coefficients of Equation 7 of DDPM paper
        self.posterior_mean_coef1 = self.betas * tf.math.sqrt(self.alpha_bar_prev) / self.one_minus_alpha_bar
        self.posterior_mean_coef2 = (1.0 - self.alpha_bar_prev) * tf.math.sqrt(self.alphas) / self.one_minus_alpha_bar

        # Loss coefficients P2 Paper: https://arxiv.org/abs/2204.00227
        self.lambda_t = ((1.0 - self.betas) * (1.0 - self.alpha_bar)) / self.betas
        self.SNR_t = self.alpha_bar / (1.0 - self.alpha_bar)
        self.lambda_t_tick = self.lambda_t / ((k + self.SNR_t) ** gamma)

        # used in combination with the simplified loss, since the weighting of L_simple is allready 1, meaning it has
        # been multiplied with lambda_t already.
        self.lambda_t_tick_simple = 1 / ((k + self.SNR_t) ** gamma)


class GaussianDiffusion:
    """Abstraction of a gausian diffusion process."""

    def __init__(
        self,
        betaSchedule: BetaSchedule,
        varianceType: str,
    ) -> None:
        """Abstraction of a gausian diffusion process.

        Args:
            betaSchedule (BetaSchedule): Beta schedule for the forward diffusion process.
            varianceType (str): ["lower_bound" | "upper_bound" | "learned" | "learned_range"].
        """
        self.bs = betaSchedule
        self.policy = tf.keras.mixed_precision.global_policy()
        self.varianceType = varianceType

    def _SampledScalarToTensor(
        self,
        array: tf.Tensor,
        index: Union[tf.Tensor, Iterable[int], int],
        shape: Tuple[int, int, int, int] = (-1, 1, 1, 1),
    ) -> tf.Tensor:
        """Takes a sample from a given `array` at a given `index` and creates a tensor of shape `shape`.

        The shape (-1,1,1,1) creates a tensor of said shape, which is then prodcasted to the shape of the operand

        Args:
            array (tf.Tensor): Array from which sample is taken.
            index (Union[tf.Tensor, Iterable[int], int]): Index at which sample is taken from `array`.
            shape (Tuple[int, int, int, int], optional): Shape of the resulting tensor. Defaults to (-1, 1, 1, 1).

        Returns:
            tf.Tensor: New tensor of shape `shape`
        """
        # for whatever reason when I reshape, the tensor is casted to float64 and then the multiplication bellow throws
        # an exception. Therefore I cast it to float32.
        return tf.cast(tf.reshape(tf.experimental.numpy.take(array, index), shape), dtype=self.policy.variable_dtype)

    def draw_random_timestep(
        self,
        num: int,
    ) -> tf.Tensor:
        """Draws a random timestep from a uniform distribution.

        Args:
            num (int): Batchsize and number of samples drawn from the distribution

        Returns:
            tf.Tensor: Random timestep
        """
        return tf.random.uniform(shape=[num], minval=0, maxval=self.bs.config["timesteps"], dtype=tf.int32)

    def q_xt_given_x0_mean_var(
        self,
        x0: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward diffusion kernel q(xt|x0).

        Args:
            x0 (tf.Tensor): Original data sample.
            t (Union[tf.Tensor, np.ndarray]): timestep

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: mean, variance and log(variance).
        """
        mean = self._SampledScalarToTensor(self.bs.sqrt_alpha_bar, t) * x0
        var = self._SampledScalarToTensor(self.bs.one_minus_alpha_bar, t)
        log_var = self._SampledScalarToTensor(self.bs.log_one_minus_alpha_bar, t)
        return mean, var, log_var

    def q_sample_xt(
        self,
        x0: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generates a sample xt ~ q(xt|x0) at timestep `t` using the reparameterization trick.

        Args:
            x0 (tf.Tensor): Original data sample
            t (Union[tf.Tensor, np.ndarray]): timestep for which noisy sample shall be created.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Noisy sample `xt` and `noise` used to construct the sample.
        """
        noise = tf.random.normal(shape=x0.shape, dtype=self.policy.variable_dtype)
        tensor_sqrt_alpha_bar_t = self._SampledScalarToTensor(self.bs.sqrt_alpha_bar, t)
        tensor_sqrt_one_minus_alpha_bar_t = self._SampledScalarToTensor(self.bs.sqrt_one_minus_alpha_bar, t)
        xt = tensor_sqrt_alpha_bar_t * x0 + tensor_sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def q_xtm1_given_x0_xt_mean_var(
        self,
        x0: tf.Tensor,
        xt: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Tractible posterior distribution q(x(t-1)|xt,x0).

        Args:
            x0 (tf.Tensor): Original data sample
            xt (tf.Tensor): Noisy data sample at timestep t.
            t (Union[tf.Tensor, np.ndarray]): timestep.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: mean, variance and log(variance).
        """
        mean = (
            self._SampledScalarToTensor(self.bs.posterior_mean_coef1, t) * x0
            + self._SampledScalarToTensor(self.bs.posterior_mean_coef2, t) * xt
        )
        var = self._SampledScalarToTensor(self.bs.posterior_var, t)
        log_var = self._SampledScalarToTensor(self.bs.posterior_log_var_clipped, t)
        return mean, var, log_var

    def p_xtm1_given_xt_mean_var(
        self,
        xt: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
        model_prediction: tf.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> Dict[str, tf.Tensor]:
        """Aproximated intractible posterior distribution p(x_{t-1}|x_t).

        Real posterior distribution q(x_{t-1}|x_t) is intractible, hence it is aproximated by a neural network that
        learnes the approximated intractible posterior distribution p(x_{t-1}|x_t).

        Args:
            xt (tf.Tensor): Noisy data sample at timestep t.
            t (Union[tf.Tensor, np.ndarray]): timestep.
            model_prediction (tf.Tensor): Output of the network predicting the noise and optionally the variance.
            clip_denoised (bool, optional): If true `pred_x0` is clipped [-1,1]. Defaults to True.
            denoised_fn (Optional[Callable[[tf.Tensor], tf.Tensor]], optional): Function to call upon a prediction of
                `x0`. Defaults to None.

        Raises:
            ValueError: if provided `self.varianceType` is not specified.

        Returns:
            Dict[str, tf.Tensor]: mean, variance and log(variance) of the approximated posterior and a prediction of
                the initial x0. Keys are `mean`, `var`, `log_var` and `pred_x0`
        """
        # Variance
        if self.varianceType == "learned_range":
            pred_noise, pred_var = tf.split(
                model_prediction, 2, axis=-1
            )  # assumes that model has two outputs when variance is learned
            # LEARN_VARIANCE learns a range, not the final value! Was proposed by improved DDPM paper
            log_lower_bound = self._SampledScalarToTensor(self.bs.posterior_log_var_clipped, t)
            log_upper_bound = self._SampledScalarToTensor(tf.math.log(self.bs.betas), t)
            # pred_var is [-1, 1] for [log_lower_bound, log_upper_bound]. -> scaling to [0,1]
            v = (pred_var + 1) / 2
            log_var = v * log_upper_bound + (1 - v) * log_lower_bound
            var = tf.math.exp(log_var)  # = Sigma_theta from the paper
        elif self.varianceType == "learned":
            pred_noise, pred_var = tf.split(
                model_prediction, 2, axis=-1
            )  # assumes that model has two outputs when variance is learned
            log_var = pred_var
            var = tf.math.exp(pred_var)  # = Sigma_theta from the paper
        elif self.varianceType == "lower_bound":
            pred_noise = model_prediction
            var = self.bs.posterior_var
            log_var = tf.math.log(self.bs.posterior_var)
        elif self.varianceType == "upper_bound":
            pred_noise = model_prediction
            var = self.bs.betas
            log_var = tf.math.log(self.bs.betas)
        else:
            raise ValueError(f"Variance type {self.varianceType} is not defined.")

        var = self._SampledScalarToTensor(var, t)
        log_var = self._SampledScalarToTensor(log_var, t)

        # predicted x0
        def _process_x0(x: tf.Tensor) -> tf.Tensor:
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return tf.clip_by_value(x, -1, 1)
            return x

        pred_x0 = _process_x0(self._predict_x0_from_eps(xt, t, pred_noise))

        # mean
        mean, _, _ = self.q_xtm1_given_x0_xt_mean_var(pred_x0, xt, t)

        return {"mean": mean, "var": var, "log_var": log_var, "pred_x0": pred_x0}

    def p_sample_xtm1_given_xt(
        self,
        xt: tf.Tensor,
        model_prediction: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
    ) -> tf.Tensor:
        """Generates a sample x_{t-1} ~ p(x_{t-1}|xt), the approximated intractible posterior distr. at timestep `t`.

        Used by the DDPM sampler.

        Args:
            xt (tf.Tensor): Noisy data sample at timestep t.
            model_prediction (tf.Tensor): Output of the network predicting the noise and optionally the variance.
            t (Union[tf.Tensor, np.ndarray]): timestep

        Returns:
            tf.Tensor: less noisy sample `x_{t-1}`
        """
        p_xtm1_given_xt = self.p_xtm1_given_xt_mean_var(xt, t, model_prediction)

        z = tf.random.normal(shape=xt.shape)
        mask = 0 if t == 0 else 1

        return p_xtm1_given_xt["mean"] + mask * tf.math.sqrt(p_xtm1_given_xt["var"]) * z

    def ddim_sample(
        self,
        xt: tf.Tensor,
        pred_noise: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
        sigma_t: tf.Tensor,
    ) -> tf.Tensor:
        """Sample x_{t-1} ~ p(x_{t-1}|xt), the approximated posterior distr. using DDIM backward diffusion process.

        Args:
            xt (tf.Tensor): Noisy data sample at timestep t.
            pred_noise (tf.Tensor): Output of the network predicting the noise
            t (Union[tf.Tensor, np.ndarray]): timestep.
            sigma_t (tf.Tensor): noise tensor.

        Returns:
            tf.Tensor: less noisy sample `x_{t-1}`
        """
        sqrt_one_minus_alpha_bar = self._SampledScalarToTensor(self.bs.sqrt_one_minus_alpha_bar, t)
        sqrt_alpha_bar = self._SampledScalarToTensor(self.bs.sqrt_alpha_bar, t)
        t = t - 1
        alpha_prev = self._SampledScalarToTensor(self.bs.alphas, t - 1)

        pred = tf.math.sqrt(alpha_prev) * (xt - (sqrt_one_minus_alpha_bar) * pred_noise) / sqrt_alpha_bar

        pred = pred + tf.math.sqrt(1 - alpha_prev - (sigma_t**2)) * pred_noise
        noise = tf.random.normal(shape=xt.shape)
        mask = 0 if t == 0 else 1
        return pred + mask * sigma_t * noise

    def _predict_x0_from_eps(
        self,
        xt: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
        eps: tf.Tensor,
    ) -> tf.Tensor:
        """Approximates the initial data sample `x0` from a noise prediction, the beta schedule and the noisy sample.

        Equation 15 DDPM paper.

        Args:
            xt (tf.Tensor): Noisy data sample at timestep t
            t (Union[tf.Tensor, np.ndarray]): timestep
            eps (tf.Tensor): Output of the network predicting the noise

        Returns:
            tf.Tensor: Approximation of x0
        """
        tensor_sqrt_recip_alpha_bar = self._SampledScalarToTensor(self.bs.sqrt_recip_alpha_bar, t)
        tensor_sqrt_recip_alpha_bar_minus_one = self._SampledScalarToTensor(self.bs.sqrt_recip_alpha_bar_minus_one, t)
        return tensor_sqrt_recip_alpha_bar * xt - tensor_sqrt_recip_alpha_bar_minus_one * eps
