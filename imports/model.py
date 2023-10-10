import logging
import math
import os
import sys
from functools import partial

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import tensorflow_addons as tfa
from einops import rearrange


# Get the parent directory to be able to import the files located in imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import imports.diffusion as cdd_diffusion
import imports.helper as cdd_helper
import imports.losses as cdd_losses


# helpers functions
def exists(x):
    return x is not None


# use default params that apply to all layers
MyConv2D = partial(tf.keras.layers.Conv2D, padding="SAME", use_bias=False, kernel_initializer="glorot_uniform")
MyDense = partial(tf.keras.layers.Dense, use_bias=True, kernel_initializer="glorot_uniform")


# We will use this to convert timestamps to time encodings
class SinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim, max_positions=10000, **kwargs):
        super(SinusoidalPosEmb, self).__init__(**kwargs)
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, **kwargs):
        x = tf.cast(x, tf.float32)
        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        return tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

    def get_config(self):
        config = super(SinusoidalPosEmb, self).get_config()
        config.update(
            {
                "dim": self.dim,
                "max_positions": self.max_positions,
            }
        )
        return config


class Identity(tf.keras.layers.Layer):
    """Layer that passes the input through without performing any operation."""

    def call(self, x, **kwargs):
        return x


class Residual(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        """Adds a residual connection to a layer.
        ``out = layer(in) + in``

          Args:
              * layer: A tensorflow layer to which the residual connection shall be applied to.
        """
        super(Residual, self).__init__(**kwargs)
        self.layer = layer

    def call(self, x, **kwargs):
        return self.layer(x, **kwargs) + x


class UpSample(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), filters=None, useConv=False, interpolation="nearest", **kwargs):
        """Upsampling layer with configurable size and interpolation method. Optionally a convolution can
        be applied to change not only the spatial width but also the number of channels.

        Args:
            * size: (size_h, size_w) horizontal and vertical size of the interpolation window.
            * filters: number of filters for the optional convolution. Defaults to None.
            * useConv: Wheather or not to apply a convolution at the end. Defaults to False.
            * interpolation: Interpolation method. Defaults to "nearest".
        """
        super(UpSample, self).__init__(**kwargs)
        self.size = size
        self.useConv = useConv
        self.interpolation = interpolation
        self.up = tf.keras.layers.UpSampling2D(size=size, interpolation=interpolation)

        if useConv:
            self.conv = MyConv2D(filters=filters, kernel_size=3)

    def call(self, x, **kwargs):
        x = self.up(x)
        if self.useConv:
            x = self.conv(x)
        return x

    def get_config(self):
        config = super(UpSample, self).get_config()
        config.update({"size": self.size, "useConv": self.useConv, "interpolation": self.interpolation})
        return config


class SuperResUpSample(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        """Downsampling layer that either uses a convolution of stride 2 or an AveragePooling2D layer.

        Args:
            * pool_size (optional): _description_. Defaults to (2,2).
            * downsampling (optional): _description_. Defaults to 'average'.
        """
        super(SuperResUpSample, self).__init__(**kwargs)
        self.size = size

    def build(self, inputShape):
        channels = inputShape[-1]  # 4: rgbd, 3: rgb , 1:depth
        self.up_rgb = None
        self.up_depth = None
        if channels == 4:
            self.data_format = "rgbd"
            self.up_rgb = tf.keras.layers.UpSampling2D(size=self.size, interpolation="bilinear")
            self.up_depth = tf.keras.layers.UpSampling2D(size=self.size, interpolation="nearest")
            self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        elif channels == 3:
            self.data_format = "rgb"
            self.up_rgb = tf.keras.layers.UpSampling2D(size=self.size, interpolation="bilinear")
        elif channels == 1:
            self.data_format = "depth"
            self.up_depth = tf.keras.layers.UpSampling2D(size=self.size, interpolation="nearest")
        else:
            raise ValueError

    def call(self, x, **kwargs):
        if self.data_format == "rgbd":
            renders, depths = tf.split(x, [3, 1], axis=-1)
            renders = self.up_rgb(renders)
            depths = self.up_depth(depths)
            x = self.concatenate([renders, depths])
        elif self.data_format == "rgb":
            x = self.up_rgb(x)
        elif self.data_format == "depth":
            x = self.up_depth(x)
        return x

    def get_config(self):
        config = super(SuperResUpSample, self).get_config()
        config.update({"size": self.size})
        return config


class DownSample(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), downsampling="average", **kwargs):
        """Downsampling layer that either uses a convolution of stride 2 or an AveragePooling2D layer.

        Args:
            * pool_size (optional): _description_. Defaults to (2,2).
            * downsampling (optional): _description_. Defaults to 'average'.
        """
        super(DownSample, self).__init__(**kwargs)
        assert downsampling in ["average", "conv", "max", "learned_pooling"]
        self.downsampling = downsampling
        self.pool_size = pool_size

    def build(self, inputShape):
        if self.downsampling == "conv":
            self.down = MyConv2D(filters=inputShape[-1], kernel_size=3, strides=2)
        elif self.downsampling == "average":
            self.down = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size)
        elif self.downsampling == "max":
            self.down = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size)
        else:
            raise Exception

    def call(self, x, **kwargs):
        return self.down(x)

    def get_config(self):
        config = super(UpSample, self).get_config()
        config.update({"pool_size": self.pool_size, "downsampling": self.downsampling})
        return config


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        """Applies LayerNormalization before calling a provided layer.

        Args:
            * layer: Layer which should be executed after the LayerNormalization
        """
        super(PreNorm, self).__init__(**kwargs)
        self.layer = layer
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, **kwargs):
        x = self.norm(x, **kwargs)
        return self.layer(x, **kwargs)


# building block modules
class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        outChannels,
        gn_group_size=8,
        dropoutRate=0.0,
        strides=1,
        kernel_size=3,
        up=False,
        down=False,
        downsampling="average",
        **kwargs,
    ):
        """Basic building block containing normalization, activation, optional dropout and a convolution.
        Optionally encoorperates a time_embedding provided as gamma_beta using AdaGN (Adaptive Group Norm)

        Args:
            * outChannels: Number of Channels in the resulting output feature map
            * gn_group_size: Groupsize for GroupNorm layer. Defaults to 32.
            * dropoutRate: Dropout rate for SpatialDropout operation. Defaults to 0.0.
            * strides: Strides of the convolution kernel. Defaults to 1.
            * kernel_size: Kernelsize of the convolution. Defaults to 3.
            * up: If true, adds spatial upsampling layer. Defaults to False.
            * down: If true, adds spatial downsampling layer. Defaults to False.
            * downsampling: Downsampling method to use Defaults to 'average'.

        """
        super(Block, self).__init__(**kwargs)
        self.outChannels = outChannels
        self.gn_group_size = gn_group_size
        self.dropoutRate = dropoutRate
        self.strides = strides
        self.kernel_size = kernel_size
        self.up = up
        self.down = down
        self.downsampling = downsampling

        if self.up:
            self.updown = UpSample()
        elif self.down:
            self.updown = DownSample(downsampling=downsampling)
        else:
            self.updown = Identity()

        self.conv = MyConv2D(filters=outChannels, kernel_size=kernel_size, strides=strides)
        self.norm = tfa.layers.GroupNormalization(gn_group_size, epsilon=1e-05)
        self.act = tf.keras.layers.Activation(tf.keras.activations.gelu)
        self.dropout = tf.keras.layers.Dropout(dropoutRate)

    def call(self, x, gamma_beta=None, **kwargs):
        x = self.conv(x)
        x = self.norm(x, **kwargs)
        if exists(gamma_beta):
            gamma, beta = gamma_beta
            x = x * (gamma + 1) + beta

        x = self.act(x)
        x = self.dropout(x, **kwargs)
        return self.updown(x)

    def get_config(self):
        config = super(Block, self).get_config()
        config.update(
            {
                "outChannels": self.outChannels,
                "gn_group_size": self.gn_group_size,
                "dropoutRate": self.dropoutRate,
                "strides": self.strides,
                "up": self.up,
                "down": self.down,
                "downsampling": self.downsampling,
            }
        )
        return config


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        outChannels,
        time_emb_dim=None,
        gn_group_size=8,
        dropoutRate=0.0,
        up=False,
        down=False,
        dropResBlockRate=0.0,
        downsampling="average",
        **kwargs,
    ):
        super(ResnetBlock, self).__init__(**kwargs)

        self.outChannels = outChannels
        self.time_emb_dim = time_emb_dim
        self.gn_group_size = gn_group_size
        self.dropoutRate = dropoutRate
        self.up = up
        self.down = down
        self.dropResBlockRate = dropResBlockRate
        self.downsampling = downsampling

        self.mlp_dense = MyDense(units=outChannels * 2)

        if self.up:
            self.updown_x = UpSample()
        elif self.down:
            self.updown_x = DownSample(downsampling=downsampling)
        else:
            self.updown_x = Identity()

        if self.up or self.down:
            self.outLayers = tf.keras.Sequential(
                [
                    tfa.layers.GroupNormalization(gn_group_size, epsilon=1e-05),
                    tf.keras.layers.Activation(tf.keras.activations.gelu),
                    tf.keras.layers.Dropout(dropoutRate),
                    MyConv2D(filters=outChannels, kernel_size=3, strides=1, padding="SAME"),
                ]
            )

        self.block1_h = Block(outChannels, gn_group_size=gn_group_size, up=up, down=down, downsampling=downsampling)
        self.block2_h = Block(outChannels, gn_group_size=gn_group_size, dropoutRate=dropoutRate)

    def build(self, inputShape):
        self.skip_x = (
            MyConv2D(filters=self.outChannels, kernel_size=3) if inputShape[-1] != self.outChannels else Identity()
        )
        super(ResnetBlock, self).build(inputShape)
        self.built = True

    def call(self, x, time_emb=None, **kwargs):
        if not kwargs["training"]:
            return self._call(x, time_emb, **kwargs)
        else:
            if np.random.binomial(n=1, p=self.dropResBlockRate):
                return x
            else:
                return self._call(x, time_emb, **kwargs)

    def _call(self, x, time_emb=None, **kwargs):
        gamma_beta = None
        if exists(self.time_emb_dim) and exists(time_emb):
            time_emb = self.mlp_dense(time_emb)
            time_emb = rearrange(time_emb, "b c -> b 1 1 c")
            gamma_beta = tf.split(time_emb, num_or_size_splits=2, axis=-1)

        h = self.block1_h(x, gamma_beta=gamma_beta, **kwargs)
        h = self.block2_h(h, **kwargs)
        x = self.updown_x(x)
        x = self.skip_x(x)
        y = h + x

        if self.up or self.down:
            y = self.outLayers(y)

        return y

    def get_config(self):
        config = super(ResnetBlock, self).get_config()
        config.update(
            {
                "outChannels": self.outChannels,
                "time_emb_dim": self.time_emb_dim,
                "gn_group_size": self.gn_group_size,
                "dropoutRate": self.dropoutRate,
                "up": self.up,
                "down": self.down,
                "dropResBlockRate": self.dropResBlockRate,
                "downsampling": self.downsampling,
            }
        )
        return config


class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, dim_head=32, **kwargs):
        super(LinearAttention, self).__init__(**kwargs)
        self.dim = dim
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

        self.attend = tf.keras.layers.Softmax()
        self.to_qkv = MyConv2D(filters=self.hidden_dim * 3, kernel_size=1)
        self.to_out = MyConv2D(filters=dim, kernel_size=1)

    def call(self, x, **kwargs):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = (rearrange(t, "b x y (h c) -> b h c (x y)", h=self.heads) for t in qkv)

        q = tf.nn.softmax(q, axis=-2)
        k = tf.nn.softmax(k, axis=-1)

        q = q * self.scale
        context = tf.einsum("b h d n, b h e n -> b h d e", k, v)

        out = tf.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b x y (h c)", h=self.heads, x=h, y=w)
        return self.to_out(out)

    def get_config(self):
        config = super(LinearAttention, self).get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_head": self.dim_head,
                "scale": self.scale,
                "heads": self.heads,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, dim_head=32, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dim = dim
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

        self.to_qkv = MyConv2D(filters=self.hidden_dim * 3, kernel_size=1)
        self.to_out = MyConv2D(filters=dim, kernel_size=1)

    def call(self, x, **kwargs):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = (rearrange(t, "b x y (h c) -> b h c (x y)", h=self.heads) for t in qkv)
        q = q * self.scale

        sim = tf.einsum("b h d i, b h d j -> b h i j", q, k)
        sim_max = tf.stop_gradient(tf.expand_dims(tf.argmax(sim, axis=-1), axis=-1))
        sim_max = tf.cast(sim_max, x.dtype)
        sim = sim - sim_max
        attn = tf.nn.softmax(sim, axis=-1)

        out = tf.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b x y (h d)", x=h, y=w)
        return self.to_out(out)

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_head": self.dim_head,
                "scale": self.scale,
                "heads": self.heads,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


class InterSkipConnection(tf.keras.layers.Layer):
    def __init__(self, pool_size, filters=64, gn_group_size=8, **kwargs):
        """Skip connection between encoder and decoder of a UNet. Can be applied between different levels of the UNet.

        Args:
            * filters (optional): Number of channels in the resulting feature map. Defaults to 64.
            * pool_size: poolsize for the MaxPool2D opperation to shrink the spatial width.
            * gn_group_size: Size of groups for the GroupNorm layer. Defaults to 32.
        """
        super(InterSkipConnection, self).__init__(**kwargs)
        self.filters = filters
        self.pool_size = pool_size
        self.gn_group_size = gn_group_size
        self.pool = DownSample((pool_size, pool_size), downsampling="max") if pool_size > 1 else Identity()
        self.block = Block(outChannels=filters, gn_group_size=gn_group_size)

    def call(self, x, **kwargs):
        x = self.pool(x)
        return self.block(x, **kwargs)

    def get_config(self):
        config = super(InterSkipConnection, self).get_config()
        config.update({"filters": self.filters, "pool_size": self.pool_size, "gn_group_size": self.gn_group_size})
        return config


class IntraSkipConnection(tf.keras.layers.Layer):
    def __init__(self, size, filters=64, gn_group_size=8, **kwargs):
        super(IntraSkipConnection, self).__init__(**kwargs)
        self.filters = filters
        self.size = size
        self.gn_group_size = gn_group_size
        self.up = UpSample(size=(size, size), interpolation="nearest") if size > 1 else Identity()
        self.block = Block(outChannels=filters, gn_group_size=gn_group_size)

    def call(self, x, **kwargs):
        x = self.up(x)
        return self.block(x, **kwargs)

    def get_config(self):
        config = super(IntraSkipConnection, self).get_config()
        config.update({"filters": self.filters, "size": self.size, "gn_group_size": self.gn_group_size})
        return config


###########################
#        UNET             #
###########################
class Unet(tf.keras.Model):
    def __init__(
        self,
        baseDim=64,
        dim_mults=(1, 2, 4, 8),
        numBlocks=(2, 2, 2, 2),
        numResBlocks=(1, 1, 1, 1),
        attentionBlocks=(1, 1, 1, 1),
        dropResBlockRate=(0.0, 0.0, 0.0, 0.0),
        diffusionChannels=1,
        gn_group_size=8,
        learned_variance=False,
        dropoutRate=0.1,
        downsampling="average",
    ):
        """Unet Model with single input and output conditioned on a timestep embedding.

        Input:
            Feature map (e.g. image, RGBD or depth map) at any diffusion timestep
            Timestep: Diffusion timestep
            Input shape: [batch, height, width, diffusionChannels], [batch]

        Output:
            if learned_variance:
              predicted Noise and variance added at the given timestep
              Output shape: [batch, height, width, 2* diffusionChannels]
            else:
              predicted Noise added at the given timestep
              Output shape: [batch, height, width, diffusionChannels]

        Args:
            * baseDim (int): Minimum number of channels that is used as a basis for each level of the UNet Model. Defaults to 64.
            * dim_mults (list[int]): List of integers. Each number represents a new level of the UNet where the number of channels is the multipliplication of baseDim with the respective multiplier. Total Number of layers of the UNet = len(dim_mults). Defaults to (1, 2, 4, 8).
            * numBlocks (list[int]): A list of integers where each element represent the number of blocks created in a stage of the UNet Model, where Block(x) = Attention(ResBlock(x)). Defaults to (2, 2, 2, 2).
            * numResBlocks (list[int]): A list of integers where each element represent the number of ResBlocks created within a Block in a stage of the UNet Model, where each Resblock contains 'numResBlocks' consecutive resBlocks. Defaults to (1, 1, 1, 1).
            * diffusionChannels (int): Number of channels of the data input that is beeing diffused. Depending on 'learned_variance' parameter, the number of output channels are calculated accordingly. Defaults to 1.
            * gn_group_size (int): Groupsize for the Groupnormalization layers within the model. Defaults to 32.
            * learned_variance (bool): The Variance of the diffusion model can eitherbe set or learned. If learned, the UNet Model predicts the noise and the variance, hence doubles the output channels. Defaults to False.
            * dropoutRate (float): Dropoutrate parameter for the SpatialDropout layers. Defaults to 0.1.
        """
        super(Unet, self).__init__()

        assert (
            len(dim_mults) == len(numBlocks) == len(numResBlocks) == len(dropResBlockRate) == len(attentionBlocks)
        ), "Parameters 'dim_mults', 'numBlocks', 'dropResBlockRate', 'numResBlocks' and 'attentionBlocks' must have same number of elements."

        self.diffusionChannels = diffusionChannels
        self.out_dim = self.diffusionChannels * (1 if not learned_variance else 2)

        init_dim = baseDim // 2
        self.init_conv = Block(init_dim, gn_group_size=gn_group_size, kernel_size=7, name="init_conv")

        dims = [init_dim, *(baseDim * m for m in dim_mults)]
        self.in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        self.time_dim = baseDim * 4

        self.time_mlp = tf.keras.Sequential(
            [
                SinusoidalPosEmb(baseDim),
                MyDense(units=self.time_dim),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                MyDense(units=self.time_dim),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
            ],
            name="timeEmbeddings",
        )

        resBlock_p = partial(
            ResnetBlock, gn_group_size=gn_group_size, dropoutRate=dropoutRate, time_emb_dim=self.time_dim
        )

        self.encoder = []
        self.decoder = []
        self.num_resolutions = len(self.in_out)

        ##############
        ## Encoder
        #############
        for ind, ([_, outChannels], n_blocks, n_resBlocks, d_rate, attnBlock) in enumerate(
            zip(self.in_out, numBlocks, numResBlocks, dropResBlockRate, attentionBlocks)
        ):
            is_last = ind >= (self.num_resolutions - 1)

            blocks = []

            # iterating over encoder blocks. Block(x) = Attention(ResBlock(x)) -> Resblock followed by attention
            for e in range(n_blocks):
                resBlocks = []

                # iterating over resblocks. Resblock contains 'numResBlocks' consecutive resBlocks.
                for r in range(n_resBlocks):
                    resBlocks.append(
                        resBlock_p(outChannels, dropResBlockRate=d_rate, name=f"e{ind}_resBlock_r{r}_b{e}")
                    )

                linAttn = (
                    Residual(PreNorm(LinearAttention(outChannels)), name=f"e{ind}_linearAttention_b{e}")
                    if attnBlock
                    else Identity(name=f"e{ind}_linearAttention_identity_b{e}")
                )
                # adding attention
                blocks.append([resBlocks, linAttn])

            # adding downsample
            self.encoder.append(
                [
                    blocks,
                    resBlock_p(outChannels, down=True, downsampling=downsampling, name=f"e{ind}_down")
                    if not is_last
                    else Identity(name=f"e{ind}_identity"),
                ]
            )

        ##############
        ## Latent
        #############
        latent_dim = dims[-1]
        self.latent_block1 = resBlock_p(latent_dim, dropResBlockRate=dropResBlockRate[-1], name="latent_resBlock_b1")
        self.latent_attn = Residual(PreNorm(Attention(latent_dim)), name="latent_attention")
        self.latent_block2 = resBlock_p(latent_dim, dropResBlockRate=dropResBlockRate[-1], name="latent_resBlock_b2")

        ##############
        ## Decoder
        #############
        for ind, ([inChannels, _], n_blocks, n_resBlocks, d_rate, attnBlock) in enumerate(
            zip(
                reversed(self.in_out[1:]),
                reversed(numBlocks),
                reversed(numResBlocks),
                reversed(dropResBlockRate),
                reversed(attentionBlocks),
            )
        ):
            is_last = ind >= (self.num_resolutions - 1)

            blocks = []

            # iterating over decoder blocks. Block(x) = Attention(ResBlock(x)) -> Resblock followed by attention
            for d in range(n_blocks):
                resBlocks = []

                # iterating over resblocks. Resblock contains 'numResBlocks' consecutive resBlocks.
                for r in range(n_resBlocks):
                    resBlocks.append(
                        resBlock_p(
                            inChannels,
                            dropResBlockRate=d_rate,
                            name=f"d{self.num_resolutions-1-ind}_resBlock_r{r}_b{d}",
                        )
                    )

                linAttn = (
                    Residual(
                        PreNorm(LinearAttention(inChannels)), name=f"d{self.num_resolutions-1-ind}_linearAttention_b{d}"
                    )
                    if attnBlock
                    else Identity(name=f"d{self.num_resolutions-1-ind}_linearAttention_identity_b{d}")
                )
                # adding attention
                blocks.append([resBlocks, linAttn])

            # adding upsample
            self.decoder.append(
                [
                    blocks,
                    resBlock_p(inChannels, up=True, name=f"d{self.num_resolutions-1-ind}_up")
                    if not is_last
                    else Identity(name=f"d{self.num_resolutions-1-ind}_identity"),
                ]
            )

        ##############
        ## Output
        #############
        self.final_conv = tf.keras.Sequential(
            [
                ResnetBlock(baseDim, gn_group_size=gn_group_size),
                MyConv2D(filters=self.out_dim, kernel_size=1, dtype=tf.float32),
            ],
            name="output",
        )

    def build(self, inputShape):
        self.inputShape = inputShape
        inp = tf.keras.layers.Input(shape=inputShape)
        t = tf.keras.layers.Input(shape=())
        self.call(inp, t, training=False)

        self.built = True

    def call(self, input, time, **kwargs):
        x = self.init_conv(input, **kwargs)
        t = self.time_mlp(time, **kwargs)

        skips = []

        # encoder contains all these blocks and it is repeated several times. h reflects the skip connection
        for blocks, downsample in self.encoder:
            for block in blocks:
                for resBlock in block[0]:  # ResnetBlocks
                    x = resBlock(x, time_emb=t, **kwargs)
                x = block[1](x, **kwargs)  # AttentionBlock

            skips.append(x)
            x = downsample(x, time_emb=t, **kwargs)

        x = self.latent_block1(x, time_emb=t, **kwargs)
        x = self.latent_attn(x, **kwargs)
        x = self.latent_block2(x, time_emb=t, **kwargs)

        for blocks, upsample in self.decoder:
            x = tf.concat([x, skips.pop()], axis=-1)
            for block in blocks:
                for resBlock in block[0]:  # ResnetBlocks
                    x = resBlock(x, time_emb=t, **kwargs)
                x = block[1](x, **kwargs)  # AttentionBlock
            x = upsample(x, time_emb=t, **kwargs)

        x = tf.concat([x, skips.pop()], axis=-1)
        return self.final_conv(x, **kwargs)

    def PlotGraph(self, fileName="model.png"):
        if self.built:
            inp = tf.keras.layers.Input(shape=self.inputShape)
            t = tf.keras.layers.Input(shape=())
            model = tf.keras.Model(inputs=[inp, t], outputs=self.call(inp, t, training=False))
            tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file=fileName)
        else:
            raise Exception("Model has not been built yet. Run build() first.")


###########################
#     ConditionalUnet     #
###########################
class ConditionalUnet(Unet):
    def __init__(
        self,
        baseDim=64,
        dim_mults=(1, 2, 4, 8),
        numBlocks=(2, 2, 2, 2),
        numResBlocks=(1, 1, 1, 1),
        attentionBlocks=(1, 1, 1, 1),
        dropResBlockRate=(0.0, 0.0, 0.0, 0.0),
        diffusionChannels=1,
        gn_group_size=8,
        learned_variance=False,
        dropoutRate=0.1,
        downsampling="average",
    ):
        super(ConditionalUnet, self).__init__(
            baseDim,
            dim_mults,
            numBlocks,
            numResBlocks,
            attentionBlocks,
            dropResBlockRate,
            diffusionChannels,
            gn_group_size,
            learned_variance,
            dropoutRate,
            downsampling,
        )

        self.conditionConcat = tf.keras.layers.Concatenate(name="conditionConcat")

    def build(self, condShape, difShape):
        self.condShape = condShape
        self.difShape = difShape
        cond = tf.keras.layers.Input(shape=condShape)
        dif = tf.keras.layers.Input(shape=difShape)
        t = tf.keras.layers.Input(shape=())
        self.call(cond, dif, t, training=False)
        self.built = True

    def call(self, inp_condition, inp_diffusion, time, **kwargs):
        conditioned_input = self.conditionConcat([inp_condition, inp_diffusion])
        return super(ConditionalUnet, self).call(conditioned_input, time, **kwargs)

    def PlotGraph(self, fileName="model.png"):
        if self.built:
            cond = tf.keras.layers.Input(shape=self.condShape)
            dif = tf.keras.layers.Input(shape=self.difShape)
            t = tf.keras.layers.Input(shape=())
            model = tf.keras.Model(inputs=[cond, dif, t], outputs=self.call(cond, dif, t, training=False))
            tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file=fileName)
        else:
            raise Exception("Model has not been built yet. Run build() first.")


###########################
#        UNET3+           #
###########################
class Unet3plus(ConditionalUnet):
    def __init__(
        self,
        baseDim=64,
        dim_mults=(1, 2, 4, 8),
        numBlocks=(2, 2, 2, 2),
        numResBlocks=(1, 1, 1, 1),
        attentionBlocks=(1, 1, 1, 1),
        dropResBlockRate=(0.0, 0.0, 0.0, 0.0),
        concat_filters=64,
        diffusionChannels=1,
        gn_group_size=8,
        learned_variance=False,
        dropoutRate=0.1,
        downsampling="average",
    ):
        super(Unet3plus, self).__init__(
            baseDim,
            dim_mults,
            numBlocks,
            numResBlocks,
            attentionBlocks,
            dropResBlockRate,
            diffusionChannels,
            gn_group_size,
            learned_variance,
            dropoutRate,
            downsampling,
        )

        resBlock_p = partial(
            ResnetBlock, gn_group_size=gn_group_size, dropoutRate=dropoutRate, time_emb_dim=self.time_dim
        )
        self.decoderFilters = concat_filters * (
            len(dim_mults) + 1
        )  # connection from each encoder level + 1 from the latent space.

        self.decoder = []

        ##############
        ## Decoder
        #############
        for ind in range(self.num_resolutions):
            # skip connections from encoder
            e_skips = []
            for s in range(self.num_resolutions - ind):
                e_skips.append(
                    InterSkipConnection(
                        filters=concat_filters,
                        pool_size=2**s,
                        gn_group_size=gn_group_size,
                        name=f"skip_e{(self.num_resolutions -1) -ind-s}_d{(self.num_resolutions -1) -ind}",
                    )
                )

            # skip connections from decoder
            d_skips = []
            for s in range(ind + 1):
                if s < 2:  # latent space has same size as first encoder block!
                    size = 2 ** (ind)
                else:
                    size = 2 ** (ind - (s - 1))
                d_skips.append(
                    IntraSkipConnection(
                        filters=concat_filters,
                        size=size,
                        gn_group_size=gn_group_size,
                        name=f"skip_d{(self.num_resolutions -1) -ind+(s+1)}_d{(self.num_resolutions -1) -ind}",
                    )
                )

            linAttn = (
                Residual(
                    PreNorm(LinearAttention(self.decoderFilters)),
                    name=f"d{(self.num_resolutions -1)-ind}_linearAttention",
                )
                if attentionBlocks[(self.num_resolutions - 1) - ind]
                else Identity(name=f"d{self.num_resolutions-1-ind}_linearAttention_identity")
            )
            self.decoder.append(
                [
                    resBlock_p(self.decoderFilters, name=f"d{(self.num_resolutions -1)-ind}_resBlock"),
                    linAttn,
                    e_skips,
                    d_skips,
                    tf.keras.layers.Concatenate(name=f"concat{(self.num_resolutions -1)-ind}"),
                ]
            )

    def call(self, inp_condition, inp_diffusion, time, **kwargs):
        conditioned_input = self.conditionConcat([inp_condition, inp_diffusion])
        x = self.init_conv(conditioned_input)
        t = self.time_mlp(time)

        e_out = []
        d_out = []

        # encoder blocks
        for blocks, downsample in self.encoder:
            for block in blocks:
                for resBlock in block[0]:  # ResnetBlocks
                    x = resBlock(x, time_emb=t, **kwargs)
                x = block[1](x, **kwargs)  # AttentionBlock
            e_out.append(x)
            x = downsample(x, time_emb=t, **kwargs)

        # Latent blocks
        x = self.latent_block1(x, time_emb=t, **kwargs)
        x = self.latent_attn(x, **kwargs)
        x = self.latent_block2(x, time_emb=t, **kwargs)
        d_out.append(x)

        # decoder blocks
        for i, (block, attn, e_skip_ops, d_skip_ops, concat) in enumerate(self.decoder):
            # get InterConnections from encoder
            e_skips = []
            for s, skip in enumerate(list(reversed(e_out))[i::]):
                e_skips.append(e_skip_ops[s](skip))

            # get IntraConnections from decoder
            d_skips = []
            for d, skip in enumerate(list(d_out)):
                d_skips.append(d_skip_ops[d](skip))

            x = concat([*e_skips, *d_skips])
            x = block(x, t, **kwargs)
            x = attn(x, **kwargs)
            d_out.append(x)

        return self.final_conv(x, **kwargs)


###########################
#  SuperResolutionUnet    #
###########################
class SuperResolutionUnet(ConditionalUnet):
    def __init__(
        self,
        baseDim=64,
        dim_mults=(1, 2, 4, 8),
        numBlocks=(2, 2, 2, 2),
        numResBlocks=(1, 1, 1, 1),
        attentionBlocks=(1, 1, 1, 1),
        dropResBlockRate=(0.0, 0.0, 0.0, 0.0),
        diffusionChannels=1,
        gn_group_size=8,
        learned_variance=False,
        dropoutRate=0.1,
        upsamplingFactor=(2, 2),
        downsampling="average",
    ):
        super(SuperResolutionUnet, self).__init__(
            baseDim,
            dim_mults,
            numBlocks,
            numResBlocks,
            attentionBlocks,
            dropResBlockRate,
            diffusionChannels,
            gn_group_size,
            learned_variance,
            dropoutRate,
            downsampling,
        )

        # rgb will be blury, hence the blur augmentation
        self.upsample = SuperResUpSample(size=upsamplingFactor)

    def call(self, low_res, high_res, time, **kwargs):
        low_res = self.upsample(low_res)
        return super(SuperResolutionUnet, self).call(low_res, high_res, time, **kwargs)


###########################
#  SuperResolutionUnet3+    #
###########################
class SuperResolutionUnet3plus(Unet3plus):
    def __init__(
        self,
        baseDim=64,
        dim_mults=(1, 2, 4, 8),
        numBlocks=(2, 2, 2, 2),
        numResBlocks=(1, 1, 1, 1),
        attentionBlocks=(1, 1, 1, 1),
        dropResBlockRate=(0.0, 0.0, 0.0, 0.0),
        concat_filters=64,
        diffusionChannels=1,
        gn_group_size=8,
        learned_variance=False,
        dropoutRate=0.1,
        upsamplingFactor=(2, 2),
        downsampling="average",
    ):
        super(SuperResolutionUnet3plus, self).__init__(
            baseDim,
            dim_mults,
            numBlocks,
            numResBlocks,
            attentionBlocks,
            dropResBlockRate,
            concat_filters,
            diffusionChannels,
            gn_group_size,
            learned_variance,
            dropoutRate,
            downsampling,
        )

        # rgb will be blury, hence the blur augmentation
        self.upsample = SuperResUpSample(size=upsamplingFactor)

    def call(self, low_res, high_res, time, **kwargs):
        low_res = self.upsample(low_res)
        return super(SuperResolutionUnet3plus, self).call(low_res, high_res, time, **kwargs)


###########################
#     DiffusionModel      #
###########################
class DiffusionModel:
    def __init__(
        self,
        model,
        varianceType,
        diffusionSteps,
        diffusionInputShapeChannels,
        diffusionInputShapeHeightWidth,
        betaScheduleType="linear",
        lossWeighting="simple",
        lambdaVLB=None,
        mixedPrecission=True,
    ):
        self.model = model
        self.varianceType = varianceType
        self.lossWeighting = lossWeighting
        self.lambdaVLB = lambdaVLB
        self.diffusionSteps = diffusionSteps
        self.mixedPrecission = mixedPrecission
        self.betaScheduleType = betaScheduleType
        self.diffusionInputShapeChannels = diffusionInputShapeChannels
        self.diffusionInputShapeHeightWidth = diffusionInputShapeHeightWidth
        self.optimizer = None
        self.strategy = tf.distribute.get_strategy()

        self.betaSchedule = cdd_diffusion.BetaSchedule(
            schedule=self.betaScheduleType, timesteps=self.diffusionSteps, k=1.0, gamma=1.0
        )
        self.diffusion = cdd_diffusion.GaussianDiffusion(betaSchedule=self.betaSchedule, varianceType=self.varianceType)

    def compile(self, optimizer):
        if self.mixedPrecission:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        self.optimizer = optimizer

    def getCurrentLearningRate(self):
        # _current_learning_rate considers also updates performed by learning rate schedules
        if self.mixedPrecission:  # LossScaleOptimizer Wrapper Object
            optimizer = self.optimizer.inner_optimizer
        else:
            optimizer = self.optimizer

        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr
        return current_lr

    @tf.function
    def train_step(self, batched_x0, globalBatchsize):
        batched_x0_condition, batched_x0_diffusion_input = batched_x0
        batch_size = batched_x0_diffusion_input.shape[0]

        timestep_values = self.diffusion.generate_timestamp(batch_size)  # 1 value for each sample in the batch

        batched_xt_diffusion_input, noise = self.diffusion.q_sample_xt(batched_x0_diffusion_input, timestep_values)

        if self.lossWeighting == "simple":
            loss_scaling_factor = 1
        elif self.lossWeighting == "P2":
            loss_scaling_factor = self.diffusion.SampledScalarToTensor(
                self.diffusion.bs.lambda_t_tick_simple, timestep_values, shape=(batch_size, 1, 1, 1)
            )
        else:
            raise Exception(f"Undefined lossWeighting provided: {self.lossWeighting}")

        with tf.GradientTape() as tape:
            prediction = self.model(batched_x0_condition, batched_xt_diffusion_input, timestep_values)

            if self.varianceType in ["learned", "learned_range"]:
                # split prediction into noise and var
                # recombine it with applying tf.stop_gradient to the noise value
                # feed it to the Get_L_VLB_Term method
                pred_noise, pred_var = tf.split(prediction, 2, axis=-1)
                pred_stopped_noise = tf.concat((tf.stop_gradient(pred_noise), pred_var), axis=-1)
                loss_simple = cdd_losses.L_simple(noise, pred_noise, globalBatchsize, loss_scaling_factor)
                loss_vlb = self.lambdaVLB * cdd_losses.L_VLB(
                    pred_stopped_noise,
                    batched_x0_diffusion_input,
                    batched_xt_diffusion_input,
                    timestep_values,
                    self.diffusion,
                    globalBatchsize,
                )
                loss = loss_simple + loss_vlb
            else:
                loss = cdd_losses.L_simple(noise, prediction, globalBatchsize, loss_scaling_factor)

            if self.mixedPrecission:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if self.mixedPrecission:
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def test_step(self, batched_x0, globalBatchsize):
        batched_x0_condition, batched_x0_diffusion_input = batched_x0
        batch_size = batched_x0_diffusion_input.shape[0]

        timestep_values = self.diffusion.generate_timestamp(batch_size)  # 1 value for each sample in the batch

        batched_xt_diffusion_input, noise = self.diffusion.q_sample_xt(batched_x0_diffusion_input, timestep_values)

        if self.lossWeighting == "simple":
            loss_scaling_factor = 1
        elif self.lossWeighting == "P2":
            loss_scaling_factor = self.diffusion.SampledScalarToTensor(
                self.diffusion.bs.lambda_t_tick_simple, timestep_values, shape=(batch_size, 1, 1, 1)
            )
        else:
            raise Exception(f"Undefined lossWeighting provided: {self.lossWeighting}")

        prediction = self.model(batched_x0_condition, batched_xt_diffusion_input, timestep_values, training=False)

        if self.varianceType in ["learned", "learned_range"]:
            # split prediction into noise and var
            # recombine it with applying tf.stop_gradient to the noise value
            # feed it to the Get_L_VLB_Term method
            pred_noise, pred_var = tf.split(prediction, 2, axis=-1)
            pred_stopped_noise = tf.concat((tf.stop_gradient(pred_noise), pred_var), axis=-1)
            loss_simple = cdd_losses.L_simple(noise, pred_noise, globalBatchsize, loss_scaling_factor)
            loss_vlb = self.lambdaVLB * cdd_losses.L_VLB(
                pred_stopped_noise,
                batched_x0_diffusion_input,
                batched_xt_diffusion_input,
                timestep_values,
                self.diffusion,
                globalBatchsize,
            )
            loss = loss_simple + loss_vlb
        else:
            loss = cdd_losses.L_simple(noise, prediction, globalBatchsize, loss_scaling_factor)

        return loss

    # @tf.function
    def eval_step(self, batched_x0, globalBatchsize, threshold=-0.9):
        batched_x0_condition, batched_x0_diffusion_input = batched_x0
        inputShape = batched_x0_diffusion_input.shape
        VLB_terms = []

        xt_diffusion_input_reverse = tf.random.normal(inputShape)

        for i in tqdm(range(self.diffusionSteps), ncols=100):
            t = np.expand_dims(np.array(self.diffusionSteps - i - 1, np.int32), 0)
            model_prediction = self.model(batched_x0_condition, xt_diffusion_input_reverse, t, training=False)
            xt_diffusion_input_forward, _ = self.diffusion.q_sample_xt(batched_x0_diffusion_input, t)
            # update xt_diffusion_input_reverse for next cycle by sampling diffusion_input from distr.
            xt_diffusion_input_reverse = self.diffusion.p_sample_xtm1_given_xt(
                xt_diffusion_input_reverse, model_prediction, t
            )

            # mean reduction of batches is performed later together with the summation of L0,L1,L2,L3 ...
            vlb_term = cdd_losses.Get_L_VLB_Term(
                model_prediction,
                batched_x0_diffusion_input,
                xt_diffusion_input_forward,
                t,
                self.diffusion,
                clip_denoised=True,
                returnBits=True,
            )
            VLB_terms.append(vlb_term)

        kl_prior = cdd_losses.Get_VLB_prior(batched_x0_diffusion_input, self.diffusion)
        VLB_terms.append(kl_prior)

        # VLB is sum of individual losses of each timestep
        # Sum all individual loss terms L0, L1, ..., LT. CAUTION: Shape of VLB_terms is (timestep, batch)! -> sum axis 0 for loss summation
        VLB = tf.math.reduce_sum(VLB_terms, axis=0)

        # reduce mean operation considering potential distributed training strategy
        if threshold is not None:
            xt_diffusion_input_reverse = tf.where(
                xt_diffusion_input_reverse < threshold, -1.0, xt_diffusion_input_reverse
            )  # Must be 1.0 not -1 so the output is not casted as int...

        # shape of iou and dice is [batch] -> one score for each batch
        metrics = cdd_helper.calc_depth_metrics(
            depth_gt=batched_x0_diffusion_input, depth_pred=xt_diffusion_input_reverse, threshold=threshold
        )
        y_shifts, x_shifts, depth_pred_shifted = cdd_helper.get_shift(
            batched_x0_diffusion_input, xt_diffusion_input_reverse, threshold=threshold
        )
        metrics_shifted = cdd_helper.calc_depth_metrics(
            depth_gt=batched_x0_diffusion_input, depth_pred=depth_pred_shifted, threshold=threshold
        )

        # perform manual loss reduction over batch axis
        VLB = (1.0 / globalBatchsize) * tf.math.reduce_sum(VLB)
        mae = (1.0 / globalBatchsize) * tf.math.reduce_sum(metrics["mae"])
        mse = (1.0 / globalBatchsize) * tf.math.reduce_sum(metrics["mse"])
        iou = (1.0 / globalBatchsize) * tf.math.reduce_sum(metrics["iou"])
        dice = (1.0 / globalBatchsize) * tf.math.reduce_sum(metrics["dice"])
        x_translation = (1.0 / globalBatchsize) * tf.math.reduce_sum(tf.math.abs(y_shifts))
        y_translation = (1.0 / globalBatchsize) * tf.math.reduce_sum(tf.math.abs(x_shifts))
        mae_shifted = (1.0 / globalBatchsize) * tf.math.reduce_sum(metrics_shifted["mae"])
        mse_shifted = (1.0 / globalBatchsize) * tf.math.reduce_sum(metrics_shifted["mse"])
        iou_shifted = (1.0 / globalBatchsize) * tf.math.reduce_sum(metrics_shifted["iou"])
        dice_shifted = (1.0 / globalBatchsize) * tf.math.reduce_sum(metrics_shifted["dice"])

        logging.info(
            f"VLB: {VLB:.4f} MAE: {mae:.4f} MSE: {mse:.4f} IoU: {iou:.4f} Dice: {dice:.4f} x_translation: {x_translation:.4f} y_translation: {y_translation:.4f} MAE shifted: {mae_shifted:.4f} MSE shifted: {mse_shifted:.4f} IoU shifted: {iou_shifted:.4f}  Dice shifted: {dice_shifted:.4f}"
        )
        logging.info("-------------------------------------------------")

        return (
            VLB,
            mae,
            mse,
            iou,
            dice,
            x_translation,
            y_translation,
            mae_shifted,
            mse_shifted,
            iou_shifted,
            dice_shifted,
        )

    @tf.function
    def distributed_train_step(self, batch_train, globalBatchsize):
        per_replica_loss = self.strategy.run(
            self.train_step,
            args=(
                batch_train,
                globalBatchsize,
            ),
        )
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    @tf.function
    def distributed_test_step(self, batch_test, globalBatchsize):
        per_replica_loss = self.strategy.run(
            self.test_step,
            args=(
                batch_test,
                globalBatchsize,
            ),
        )
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    # @tf.function
    def distributed_eval_step(self, batch_test, globalBatchsize):
        per_replica_metric_vector = self.strategy.run(
            self.eval_step,
            args=(
                batch_test,
                globalBatchsize,
            ),
        )

        # The loop iterates over loss values, not over replicas!!!
        # i.e. eval_step() returns [metric1, metric2, metric3] from each replica, so each loss element in the
        # final output is obtained by summing metric1 from all replicas, summing metric2 from all replicas, etc.
        reduced_metric_vector = []
        for per_replica_metric in per_replica_metric_vector:
            reduced_metric_vector.append(
                self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_metric, axis=None)
            )

        return reduced_metric_vector

    def DDPMSampler(self, x0_condition, sampling_steps, frames_to_output=None, threshold=-0.9):
        if frames_to_output:
            frames_to_output = min(frames_to_output, sampling_steps)
            img_list = []
            timestep_list = []
            samplesToPlot = np.linspace(0, self.diffusionSteps, frames_to_output, dtype=np.int32)
        else:
            samplesToPlot = []
        output = {}
        batch_size = x0_condition.shape[0]

        xt_diffusion_input = tf.random.normal(
            (batch_size, *list(self.diffusionInputShapeHeightWidth), self.diffusionInputShapeChannels)
        )

        for i in tqdm(
            np.linspace(0, self.diffusionSteps - 1, sampling_steps, dtype=np.int32), ncols=100, desc="DDPM sampling"
        ):
            if i in samplesToPlot:
                img_list.append(xt_diffusion_input)
                timestep_list.append(self.diffusionSteps - i)
            t = np.expand_dims(np.array(self.diffusionSteps - i - 1, np.int32), 0)
            model_prediction = self.model(x0_condition, xt_diffusion_input, t, training=False)
            xt_diffusion_input = self.diffusion.p_sample_xtm1_given_xt(xt_diffusion_input, model_prediction, t)

        if threshold is not None:
            output["x0"] = np.where(xt_diffusion_input < threshold, -1.0, xt_diffusion_input)
        else:
            output["x0"] = xt_diffusion_input

        if frames_to_output:
            img_list.append(output["x0"])
            timestep_list.append(0)
            output["xt_frames"] = tf.einsum("fbhwc->bfhwc", tf.convert_to_tensor(img_list))  # swap batch and frame axis
            output["t_frames"] = timestep_list

        return output

    def DDIMSampler(self, x0_condition, sampling_steps=50, frames_to_output=None, threshold=-0.9):
        # Define number of inference loops to run
        output = {}

        sampling_steps = sampling_steps if self.diffusionSteps >= sampling_steps else self.diffusionSteps

        batch_size = x0_condition.shape[0]

        # Create a range of inference steps that the output should be sampled at
        inference_range = range(0, self.diffusionSteps, self.diffusionSteps // sampling_steps)

        xt_diffusion_input = tf.random.normal(
            (batch_size, *list(self.diffusionInputShapeHeightWidth), self.diffusionInputShapeChannels)
        )

        if frames_to_output:
            frames_to_output = min(frames_to_output, sampling_steps)
            img_list = []
            timestep_list = []

        # Iterate over samplesteps
        for i, step in enumerate(
            tqdm(
                np.linspace(0, self.diffusionSteps - 1, sampling_steps, dtype=np.int32), ncols=100, desc="DDIM sampling"
            )
        ):
            if frames_to_output:
                img_list.append(xt_diffusion_input)
                timestep_list.append(self.diffusionSteps - step)
            t = np.expand_dims(inference_range[i], 0)
            model_prediction = self.model(x0_condition, xt_diffusion_input, t, training=False)

            if self.varianceType in ["learned", "learned_range"]:
                pred_noise, _ = tf.split(model_prediction, 2, axis=-1)
            else:
                pred_noise = model_prediction

            xt_diffusion_input = self.diffusion.ddim_sample(xt_diffusion_input, pred_noise, t, 0)

        if threshold is not None:
            output["x0"] = np.where(xt_diffusion_input < threshold, -1.0, xt_diffusion_input)
        else:
            output["x0"] = xt_diffusion_input

        if frames_to_output:
            img_list.append(output["x0"])
            timestep_list.append(0)
            output["xt_frames"] = tf.einsum("fbhwc->bfhwc", tf.convert_to_tensor(img_list))  # swap batch and frame axis
            output["t_frames"] = timestep_list

        return output
