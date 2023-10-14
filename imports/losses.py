import os
import sys
from typing import Union

import numpy as np
import tensorflow as tf

# Get the parent directory to be able to import the files located in imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from imports.diffusion import GaussianDiffusion


def ApproxStandardNormalCDF(x: tf.Tensor) -> tf.Tensor:
    """A fast approximation of the cumulative distribution function of the standard normal.

    Args:
        x (tf.Tensor): Standard normal distribution.

    Returns:
        tf.Tensor: Approximation of the cumulative distribution function of the standard normal.
    """
    return 0.5 * (1.0 + tf.math.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))


def DiscretizedGaussian_LogLikelihood(
    xt: tf.Tensor,
    mean: tf.Tensor,
    log_std: tf.Tensor,
    returnBits: bool = True,
) -> tf.Tensor:
    """Calculates the Log Likelihood of a discretized Gaussian.

    Args:
        xt (tf.Tensor): Sample x at timestep t.
        mean (tf.Tensor): Mean value of the sample x at timestep t.
        log_std (tf.Tensor): Standard deviation in logarithmic scale of the sample x at timestep t.
        returnBits (bool, optional): If true, loglikelihood is given in bits, otherwise nats. Defaults to True.

    Returns:
        tf.Tensor: Log likelihood of a discretized Gaussian in either nats or bits.
    """
    # 1. Obtain the standard normal distribution by subtracting the mean and deviding by the standard normal distribution (mean = 0, std = 1)
    # 1.1 center xt arround 0
    centered_xt = xt - mean

    # 1.2. calc inverse of the standard deviation: 1/std. Remember log_scale and log arithmetic
    inv_std = tf.math.exp(-log_std)

    # 1.3. calculating the standard normal distribution considering the discretizing error by adding/subtracting a signle bit to the cented image and scaling it by the inverse standard deviation
    plus_in = inv_std * (centered_xt + 1 / 255.0)
    min_in = inv_std * (centered_xt - 1 / 255.0)

    # 2. calculate the CDF for the standard normal distributions discretization error
    cdf_plus = ApproxStandardNormalCDF(plus_in)
    cdf_min = ApproxStandardNormalCDF(min_in)
    cdf_delta = cdf_plus - cdf_min

    # 3 clip the CDFs to a minimum value and maximum value and calculate its log
    log_cdf_plus = tf.math.log(tf.clip_by_value(cdf_plus, clip_value_min=1e-12, clip_value_max=tf.float32.max))
    log_one_minus_cdf_min = tf.math.log(
        tf.clip_by_value((1 - cdf_min), clip_value_min=1e-12, clip_value_max=tf.float32.max)
    )
    log_cdf_delta = tf.math.log(tf.clip_by_value(cdf_delta, clip_value_min=1e-12, clip_value_max=tf.float32.max))

    # 4. calculate the log probs.
    # if (xt < -0.999) use cdf_log_plus.
    # if (-0.999 < xt < 0.999) use log_cdf_delta
    # if (xt > 0.999) use log_one_minus_cdf_min
    log_probs = tf.where(xt < -0.999, log_cdf_plus, tf.where(xt > 0.999, log_one_minus_cdf_min, log_cdf_delta))

    # reduce all non-batch dimensions
    # log_probs = tf.math.reduce_mean(log_probs,axis = list(range(1,len(log_probs.shape))))
    shape = tf.shape(log_probs)
    axis_to_reduce = list(range(1, len(shape)))
    log_probs = tf.math.reduce_mean(log_probs, axis=axis_to_reduce)

    if returnBits:
        log_probs = log_probs / tf.math.log(2.0)

    return log_probs


def UnivariateNormal_KLDivergence(
    mean1: tf.Tensor,
    var1: tf.Tensor,
    mean2: tf.Tensor,
    var2: tf.Tensor,
    varianceIsLogarithmic: bool = True,
    returnBits: bool = True,
) -> tf.Tensor:
    """Calculates the Kullback Leibler Divergence between two univariate Normal distributions.

    Args:
        mean1 (tf.Tensor): Mean of the first distribution.
        var1 (tf.Tensor): Variance of the first distribution.
        mean2 (tf.Tensor): Mean value of the second distribution.
        var2 (tf.Tensor): Mean value of the first distribution.
        varianceIsLogarithmic (bool, optional): If true, indicates that provided variances are in logaritmic scale.
            Otherwise they are not. Defaults to True.
        returnBits (bool, optional): If true, loglikelihood is given in bits, otherwise nats. Defaults to True.

    Returns:
        tf.Tensor: Kullback Leibler Divergence between two univariate Normal distributions.
    """
    if varianceIsLogarithmic:
        kl_divergence = 0.5 * (
            var2 - var1 + tf.math.exp(var1 - var2) + ((mean1 - mean2) ** 2) * tf.math.exp(-var2) - 1.0
        )
    else:
        kl_divergence = tf.math.log((var2 / var1) ** 0.5) + (var1 + ((mean1 - mean2) ** 2)) / (2 * var2) - 0.5

    # reduce all non-batch dimensions
    shape = tf.shape(kl_divergence)
    axis_to_reduce = list(range(1, len(shape)))
    kl_divergence = tf.math.reduce_mean(kl_divergence, axis=axis_to_reduce)

    if returnBits:
        kl_divergence = kl_divergence / tf.math.log(2.0)

    return kl_divergence


def L_simple(
    real: tf.Tensor,
    generated: tf.Tensor,
    globalBatchsize: int,
    scaling: float = 1.0,
) -> tf.Tensor:
    """Simplified loss objective as introduced by DDPM paper.

    Args:
        real (tf.Tensor): Actual noise added to a sample in the forward diffusion.
        generated (tf.Tensor): Predicted Noise added to a sample.
        globalBatchsize (int): Batch size considering all workers running in parallel in a data parallel setup.
        scaling (float, optional): Scales the loss. If 1 results in the L_simple. Defaults to 1.0.

    Returns:
        tf.Tensor: Loss between real and generated data.
    """
    loss = scaling * (real - generated) ** 2

    # shape of loss: [batch, height, width, channel]
    loss = tf.math.reduce_mean(loss, axis=[1, 2, 3])  # mean reduce each batch individually

    # manually mean reduce by global batchsize to enable multi-worker training e.g. multi-GPU
    loss = tf.math.reduce_sum(loss)
    loss *= 1.0 / globalBatchsize

    return loss


def L_VLB(
    prediction: tf.Tensor,
    batched_x0_depth: tf.Tensor,
    batched_xt_depth: tf.Tensor,
    timestep_values: Union[tf.Tensor, np.ndarray],
    diffusion: GaussianDiffusion,
    globalBatchsize: int,
) -> tf.Tensor:
    """Calculates the Variational Lower Bound loss term for a given timestep as used in improved DDPM.

    L = Lsimple + L_VLB.

    Args:
        prediction (tf.Tensor): Predicted noise of the noise prediction model.
        batched_x0_depth (tf.Tensor): Unnoisy data at timestep 0.
        batched_xt_depth (tf.Tensor): Noisy data at timestep t obtained in the forward diffusion process.
        timestep_values (Union[tf.Tensor, np.ndarray]): Indicies of the timesteps used to obtain the actual value from
            the beta schedule.
        diffusion (GaussianDiffusion): Abstraction of the diffusion process.
        globalBatchsize (int): Batch size considering all workers running in parallel in a data parallel setup.

    Returns:
        tf.Tensor: Variational lower bound loss term for a given timestep
    """
    loss = Get_L_VLB_Term(
        prediction, batched_x0_depth, batched_xt_depth, timestep_values, diffusion, clip_denoised=False
    )  # shape: (batch,)

    # shape of loss: [batch,] -> no reduce mean required as in L_simple
    loss = tf.math.reduce_sum(loss)
    loss *= 1.0 / globalBatchsize
    return loss


# in contrast to OpenAI, I don't apply the mean_flat, since I'll do reduce_mean somewhen later
def Get_L_VLB_Term(
    model_prediction: tf.Tensor,
    x0: tf.Tensor,
    xt: tf.Tensor,
    t: Union[tf.Tensor, np.ndarray],
    diffusion: GaussianDiffusion,
    clip_denoised: bool = True,
    returnBits: bool = True,
) -> tf.Tensor:
    """Calculates the individual terms of L_VLB of a given timestep. LVLB = L0 + Lt + ... + LT.

    Args:
        model_prediction (tf.Tensor): Predicted noise of the noise prediction model.
        x0 (tf.Tensor): Unnoisy data at timestep 0.
        xt (tf.Tensor): Noisy data at timestep t obtained in the forward diffusion process.
        t (Union[tf.Tensor, np.ndarray]): Indicies of the timesteps used to obtain the actual value from the beta schedule.
        diffusion (GaussianDiffusion): Abstraction of the diffusion process.
        clip_denoised (bool, optional): _description_. Defaults to True.
        returnBits (bool, optional): If true, loglikelihood is given in bits, otherwise nats. Defaults to True.

    Returns:
        tf.Tensor: Current VLB term Lt for a given timestep t.
    """
    true_mean, _, true_log_var_clipped = diffusion.q_xtm1_given_x0_xt_mean_var(x0, xt, t)
    p_xtm1_given_xt = diffusion.p_xtm1_given_xt_mean_var(xt, t, model_prediction, clip_denoised=clip_denoised)

    # log_var*0.5 is the same as sqrt(var), which is the standard deviation!
    dg_ll = -DiscretizedGaussian_LogLikelihood(
        x0, p_xtm1_given_xt["mean"], 0.5 * p_xtm1_given_xt["log_var"], returnBits
    )

    kl = UnivariateNormal_KLDivergence(
        true_mean,
        true_log_var_clipped,
        p_xtm1_given_xt["mean"],
        p_xtm1_given_xt["log_var"],
        varianceIsLogarithmic=True,
        returnBits=returnBits,
    )

    # tf.where supports batched data. if t==0, log-likelihood is returned, otherwise the KL-divergence
    return tf.where((t == 0), dg_ll, kl)


def Get_VLB_prior(
    x0: tf.Tensor,
    diffusion: GaussianDiffusion,
    returnBits: bool = True,
) -> tf.Tensor:
    """Get the prior KL term for the variational lower-bound.

    This term can't be optimized, as it only depends on the encoder.

    Args:
        x0 (tf.Tensor): Unnoisy data at timestep 0.
        diffusion (GaussianDiffusion): Abstraction of the diffusion process.
        returnBits (bool, optional): If true, loglikelihood is given in bits, otherwise nats. Defaults to True.

    Returns:
        tf.Tensor: Prior KL term for the variational lower-bound
    """
    t = np.expand_dims(np.array(diffusion.bs.config["timesteps"] - 1, np.int32), 0)
    qt_mean, _, qt_log_variance = diffusion.q_xt_given_x0_mean_var(x0, t)
    return UnivariateNormal_KLDivergence(
        mean1=qt_mean, var1=qt_log_variance, mean2=0.0, var2=0.0, varianceIsLogarithmic=True, returnBits=returnBits
    )
