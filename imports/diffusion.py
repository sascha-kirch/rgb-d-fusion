import math

import numpy as np
import tensorflow as tf


class BetaSchedule:
    # Inspired by:
    # https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L18-L62
    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].
        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                          prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return tf.constant(np.array(betas), dtype=self.policy.variable_dtype)

    def get_beta_schedule(self, schedule, timesteps, start, stop):
        if schedule == "linear":
            betas = tf.linspace(start, stop, timesteps)
        elif schedule == "sigmoid":
            betas = tf.linspace(-6, 6, timesteps)
            betas = tf.math.sigmoid(betas) * (stop - start) + start
        elif schedule == "cosine":
            s = 0.008
            betas = self.betas_for_alpha_bar(
                timesteps,
                lambda t: tf.math.cos((t + s) / (1 + s) * math.pi / 2) ** 2,
            )
        else:
            raise Exception(f"Schedule {schedule} is not defined.")

        return tf.cast(betas, dtype=self.policy.variable_dtype)

    def __init__(self, schedule="linear", timesteps=1000, start=1e-4, stop=2e-2, k=1.0, gamma=1.0):
        self.config = {"schedule": schedule, "timesteps": timesteps, "start": start, "stop": stop}
        self.policy = tf.keras.mixed_precision.global_policy()
        self.betas = self.get_beta_schedule(schedule, timesteps, start, stop)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = tf.math.cumprod(self.alphas, 0)
        self.alpha_bar_prev = tf.concat(
            (tf.constant(np.array([1.0]), dtype=self.policy.variable_dtype), self.alpha_bar[:-1]), axis=0
        )  # shifts all elements by 1 to the right and adds 1 at the first position. last element is dropped due to [:-1]
        self.alpha_bar_next = tf.concat(
            (self.alpha_bar[1:], tf.constant(np.array([0.0]), dtype=self.policy.variable_dtype)), axis=0
        )  # shifts all elements by 1 to the right and adds 1 at the first position. last element is dropped due to [:-1]

        # Pre-calculated alpha coefficients
        self.one_minus_alpha_bar = 1.0 - self.alpha_bar
        self.sqrt_alpha_bar = tf.math.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = tf.math.sqrt(self.one_minus_alpha_bar)
        self.log_one_minus_alpha_bar = tf.math.log(self.one_minus_alpha_bar)
        self.sqrt_recip_alpha_bar = tf.math.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recip_alpha_bar_minus_one = tf.math.sqrt(1.0 / self.alpha_bar - 1)

        # Posterior Calculations q(x_{t-1} | xt, x0)
        self.posterior_var = self.betas * (1.0 - self.alpha_bar_prev) / self.one_minus_alpha_bar
        self.posterior_log_var_clipped = tf.math.log(
            tf.concat((self.posterior_var[1:2], self.posterior_var[1:]), axis=0)
        )  # [1:2] returs the element 1 in form of an tensor keeping the dimensions
        # Coefficients of Equation 7 of DDPM paper
        self.posterior_mean_coef1 = self.betas * tf.math.sqrt(self.alpha_bar_prev) / self.one_minus_alpha_bar
        self.posterior_mean_coef2 = (1.0 - self.alpha_bar_prev) * tf.math.sqrt(self.alphas) / self.one_minus_alpha_bar

        # Loss coefficients P2 Paper: https://arxiv.org/abs/2204.00227
        self.lambda_t = ((1.0 - self.betas) * (1.0 - self.alpha_bar)) / self.betas
        self.SNR_t = self.alpha_bar / (1.0 - self.alpha_bar)
        self.lambda_t_tick = self.lambda_t / ((k + self.SNR_t) ** gamma)

        # used in combination with the simplified loss, since the weighting of L_simple is allready 1, meaning it has been multiplied with lambda_t already.
        self.lambda_t_tick_simple = 1 / ((k + self.SNR_t) ** gamma)


class GaussianDiffusion:
    def __init__(self, betaSchedule, varianceType):
        self.bs = betaSchedule
        self.policy = tf.keras.mixed_precision.global_policy()
        self.varianceType = varianceType

    def SampledScalarToTensor(
        self, array, index, shape=(-1, 1, 1, 1)
    ):  # (-1,1,1,1) creates a tensor of said shape, which is then prodcasted to the shape of the operand
        """
        takes a sample at index from array and creates a tensor of shape=shape
        """
        # for whatever reason when I reshape, the tensor is casted to float64 and then the multiplication bellow throws an exception. Therefore I cast it to float32.
        return tf.cast(tf.reshape(tf.experimental.numpy.take(array, index), shape), dtype=self.policy.variable_dtype)

    def generate_timestamp(self, num):
        """
        num: batchsize and number of samples drawn from the distribution
        """
        return tf.random.uniform(shape=[num], minval=0, maxval=self.bs.config["timesteps"], dtype=tf.int32)

    def q_xt_given_x0_mean_var(self, x0, t):
        """
        returns mean and variance of q(xt|x0)
        """
        mean = self.SampledScalarToTensor(self.bs.sqrt_alpha_bar, t) * x0
        var = self.SampledScalarToTensor(self.bs.one_minus_alpha_bar, t)
        log_var = self.SampledScalarToTensor(self.bs.log_one_minus_alpha_bar, t)
        return mean, var, log_var

    def q_sample_xt(self, x0, t):
        """
        Forward Diffusion:
        sample xt from q(xt|x0)
        """
        noise = tf.random.normal(shape=x0.shape, dtype=self.policy.variable_dtype)
        tensor_sqrt_alpha_bar_t = self.SampledScalarToTensor(self.bs.sqrt_alpha_bar, t)
        tensor_sqrt_one_minus_alpha_bar_t = self.SampledScalarToTensor(self.bs.sqrt_one_minus_alpha_bar, t)
        xt = tensor_sqrt_alpha_bar_t * x0 + tensor_sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def q_xtm1_given_x0_xt_mean_var(self, x0, xt, t):
        """
        returns mean and variance of the posterior distribution q(x(t-1)|xt,x0)
        """
        mean = (
            self.SampledScalarToTensor(self.bs.posterior_mean_coef1, t) * x0
            + self.SampledScalarToTensor(self.bs.posterior_mean_coef2, t) * xt
        )
        var = self.SampledScalarToTensor(self.bs.posterior_var, t)
        log_var = self.SampledScalarToTensor(self.bs.posterior_log_var_clipped, t)
        return mean, var, log_var

    def p_xtm1_given_xt_mean_var(self, xt, t, model_prediction, clip_denoised=True, denoised_fn=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x, x0.
        """

        # Variance
        if self.varianceType == "learned_range":
            pred_noise, pred_var = tf.split(
                model_prediction, 2, axis=-1
            )  # assumes that model has two outputs when variance is learned
            # LEARN_VARIANCE learns a range, not the final value! Was proposed by improved DDPM paper
            log_lower_bound = self.SampledScalarToTensor(self.bs.posterior_log_var_clipped, t)
            log_upper_bound = self.SampledScalarToTensor(tf.math.log(self.bs.betas), t)
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
            raise Exception(f"Variance type {self.varianceType} is not defined.")

        var = self.SampledScalarToTensor(var, t)
        log_var = self.SampledScalarToTensor(log_var, t)

        # predicted x0
        def process_x0(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return tf.clip_by_value(x, -1, 1)
            return x

        pred_x0 = process_x0(self._predict_x0_from_eps(xt, t, pred_noise))

        # mean
        mean, _, _ = self.q_xtm1_given_x0_xt_mean_var(pred_x0, xt, t)

        return {"mean": mean, "var": var, "log_var": log_var, "pred_x0": pred_x0}

    def p_sample_xtm1_given_xt(self, xt, model_prediction, t):
        """
        Sample x_{t-1} from p(x_{t-1}|xt) using DDPM backward diffusion process
        """
        p_xtm1_given_xt = self.p_xtm1_given_xt_mean_var(xt, t, model_prediction)

        z = tf.random.normal(shape=xt.shape)
        mask = 0 if t == 0 else 1

        return p_xtm1_given_xt["mean"] + mask * tf.math.sqrt(p_xtm1_given_xt["var"]) * z

    def ddim_sample(self, xt, pred_noise, t, sigma_t):
        """
        Sample x_{t-1} from p(x_{t-1}|xt) using DDIM backward diffusion process
        """
        sqrt_one_minus_alpha_bar = self.SampledScalarToTensor(self.bs.sqrt_one_minus_alpha_bar, t)
        sqrt_alpha_bar = self.SampledScalarToTensor(self.bs.sqrt_alpha_bar, t)
        alpha_prev = self.SampledScalarToTensor(self.bs.alphas, t - 1)

        pred = tf.math.sqrt(alpha_prev) * (xt - (sqrt_one_minus_alpha_bar) * pred_noise) / sqrt_alpha_bar

        pred = pred + tf.math.sqrt(1 - alpha_prev - (sigma_t**2)) * pred_noise
        noise = tf.random.normal(shape=xt.shape)
        mask = 0 if t == 0 else 1
        return pred + mask * sigma_t * noise

    def _predict_x0_from_eps(self, xt, t, eps):
        """
        equation 15 DDPM paper.
        """
        tensor_sqrt_recip_alpha_bar = self.SampledScalarToTensor(self.bs.sqrt_recip_alpha_bar, t)
        tensor_sqrt_recip_alpha_bar_minus_one = self.SampledScalarToTensor(self.bs.sqrt_recip_alpha_bar_minus_one, t)
        return tensor_sqrt_recip_alpha_bar * xt - tensor_sqrt_recip_alpha_bar_minus_one * eps
