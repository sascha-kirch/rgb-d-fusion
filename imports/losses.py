import numpy as np
import tensorflow as tf


def ApproxStandardNormalCDF(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + tf.math.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))


def DiscretizedGaussian_LogLikelihood(xt, mean, log_std, returnBits=True):
    """
    :param xt: images rescaled to [-1:1] from uint8 [0,255]
    :param mean: mean of the gausian
    :param log_std: log scale standard deviation

    :return: log probabilities in nats
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


def UnivariateNormal_KLDivergence(mean1, var1, mean2, var2, varianceIsLogarithmic=True, returnBits=True):
    """
    varianceIsLogarithmic: set true, if variance is provided in log-scale.
    returnBits: if true, result is returned in bits, otherwise in nats
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


def L_simple(real, generated, globalBatchsize, scaling=1):
    loss = scaling * (real - generated) ** 2

    # shape of loss: [batch, height, width, channel]
    loss = tf.math.reduce_mean(loss, axis=[1, 2, 3])  # mean reduce each batch individually
    loss = tf.math.reduce_sum(
        loss
    )  # manually mean reduce by glubal batchsize to enable multi-worker training e.g. multi-GPU
    loss *= 1.0 / globalBatchsize

    return loss


def L_VLB(prediction, batched_x0_depth, batched_xt_depth, timestep_values, diffusion, globalBatchsize):
    loss = Get_L_VLB_Term(
        prediction, batched_x0_depth, batched_xt_depth, timestep_values, diffusion, clip_denoised=False
    )  # shape: (batch,)

    # shape of loss: [batch,] -> no reduce mean required as in L_simple
    loss = tf.math.reduce_sum(loss)
    loss *= 1.0 / globalBatchsize
    return loss


# in contrast to OpenAI, I don't apply the mean_flat, since I'll do reduce_mean somewhen later
def Get_L_VLB_Term(model_prediction, x0, xt, t, diffusion, clip_denoised=True, returnBits=True):
    """
    Get the correct term for L_VLB given a timestep t.
    LVLB = L0 + Lt + ... + LT
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


def Get_VLB_prior(x0, diffusion, returnBits=True):
    t = np.expand_dims(np.array(diffusion.bs.config["timesteps"] - 1, np.int32), 0)
    qt_mean, _, qt_log_variance = diffusion.q_xt_given_x0_mean_var(x0, t)
    return UnivariateNormal_KLDivergence(
        mean1=qt_mean, var1=qt_log_variance, mean2=0.0, var2=0.0, varianceIsLogarithmic=True, returnBits=returnBits
    )
