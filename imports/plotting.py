import io
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import signal

from PIL import Image


def smooth1d(x, window_len):
    # copied from https://matplotlib.org/stable/gallery/misc/demo_agg_filter.html
    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1 : -window_len + 1]


def smooth2d(A, sigma=3):
    # copied from https://matplotlib.org/stable/gallery/misc/demo_agg_filter.html
    window_len = max(int(sigma), 3) * 2 + 1
    A = np.apply_along_axis(smooth1d, 0, A, window_len)
    return np.apply_along_axis(smooth1d, 1, A, window_len)


class BaseFilter:
    # copied from https://matplotlib.org/stable/gallery/misc/demo_agg_filter.html
    def get_pad(self, dpi):
        return 0

    def process_image(self, padded_src, dpi):
        raise NotImplementedError("Should be overridden by subclasses")

    def __call__(self, im, dpi):
        pad = self.get_pad(dpi)
        padded_src = np.pad(im, [(pad, pad), (pad, pad), (0, 0)], "constant")
        tgt_image = self.process_image(padded_src, dpi)
        return tgt_image, -pad, -pad


class OffsetFilter(BaseFilter):
    # copied from https://matplotlib.org/stable/gallery/misc/demo_agg_filter.html
    def __init__(self, offsets=(0, 0)):
        self.offsets = offsets

    def get_pad(self, dpi):
        return int(max(self.offsets) / 72 * dpi)

    def process_image(self, padded_src, dpi):
        ox, oy = self.offsets
        a1 = np.roll(padded_src, int(ox / 72 * dpi), axis=1)
        return np.roll(a1, -int(oy / 72 * dpi), axis=0)


class GaussianFilter(BaseFilter):
    """Simple Gaussian filter."""

    # copied from https://matplotlib.org/stable/gallery/misc/demo_agg_filter.html
    def __init__(self, sigma, alpha=0.5, color=(0, 0, 0)):
        self.sigma = sigma
        self.alpha = alpha
        self.color = color

    def get_pad(self, dpi):
        return int(self.sigma * 3 / 72 * dpi)

    def process_image(self, padded_src, dpi):
        tgt_image = np.empty_like(padded_src)
        tgt_image[:, :, :3] = self.color
        tgt_image[:, :, 3] = smooth2d(padded_src[:, :, 3] * self.alpha, self.sigma / 72 * dpi)
        return tgt_image


class DropShadowFilter(BaseFilter):
    # copied from https://matplotlib.org/stable/gallery/misc/demo_agg_filter.html
    def __init__(self, sigma, alpha=0.3, color=(0, 0, 0), offsets=(0, 0)):
        self.gauss_filter = GaussianFilter(sigma, alpha, color)
        self.offset_filter = OffsetFilter(offsets)

    def get_pad(self, dpi):
        return max(self.gauss_filter.get_pad(dpi), self.offset_filter.get_pad(dpi))

    def process_image(self, padded_src, dpi):
        t1 = self.gauss_filter.process_image(padded_src, dpi)
        return self.offset_filter.process_image(t1, dpi)


# Save a GIF using logged images
def save_gif_from_array(img_list, channels, path="", interval=200):
    # Transform images from [-1,1] to [0, 255]
    # image list is of shape(frame,batch,h,w,channels)

    if channels == 1:
        img_list = np.squeeze(img_list, -1)  # new shape(frame,h,w)

    imgs = []
    for im in img_list:
        im = np.array(im)
        im = (im + 1) * 127.5
        im = np.clip(im, 0, 255).astype(np.uint8)
        im = Image.fromarray(im)
        imgs.append(im)

    imgs = iter(imgs)

    # Extract first image from iterator
    img = next(imgs)

    # Append the other images and save as GIF
    img.save(fp=path, format="GIF", append_images=imgs, save_all=True, duration=interval, loop=0)


def save_gif_from_figure_images(img_list, path="", interval=200):
    # images are in range [0,255]
    # image list is of shape(frame,batch,h,w,channels)

    img_list = np.squeeze(img_list, 1)  # new shape(frame,h,w,channels)

    imgs = []
    for im in img_list:
        im = np.array(im)
        im = np.clip(im, 0, 255).astype(np.uint8)
        im = Image.fromarray(im)
        imgs.append(im)

    imgs = iter(imgs)

    # Extract first image from iterator
    img = next(imgs)

    # Append the other images and save as GIF
    img.save(fp=path, format="GIF", append_images=imgs, save_all=True, duration=interval, loop=0)


# Save a GIF using logged images
def save_gif_from_images(img_list, channels, path="", interval=200):
    img, *imgs = [Image.open(f) for f in img_list]

    # Append the other images and save as GIF
    img.save(fp=path, format="GIF", append_images=imgs, save_all=True, duration=interval, loop=0)


def PlotSample(img, ax, title=None, returnFigure=False, threshold=-1):
    """
    img is expected to be:
      - scaled to [-1,1]
      - of shape (1, h, w, channels)
    """

    img = np.where(img <= threshold, np.nan, img)

    if len(img.shape) == 4:  # (batch, height, width, channels)
        img = np.squeeze(img, axis=0)  # squeeze batch dimension to obtain shape (h,w,channels)
    if title:
        ax.set_title(title)
    if img.shape[-1] == 1:
        ax.imshow(np.squeeze(img, -1), cmap="viridis")  # squeeze last channel to obtain shape (h,w,)
    else:
        unscaled_img = np.array((img + 1) * 127.5, np.uint8)  # values between [0,255]
        ax.imshow(unscaled_img)


def PlotDepthMap(depth, threshold, fileName=None, returnFigure=False):
    output = {}
    fig = plt.figure(figsize=(5, 5))

    if len(depth.shape) == 4:  # (batch, height, width, channels)
        depth = np.squeeze(depth, axis=0)  # squeeze batch dimension to obtain shape (h,w,channels)

    # apply threshold
    depth = np.where(depth <= threshold, -1, np.where(depth > 1, 1, depth))
    depth = np.where(depth <= threshold, np.nan, depth)

    # rescale from [-1,1] to [0:1]
    depth = (depth + 1) / 2

    plt.imshow(np.squeeze(depth, -1), cmap="viridis")  # squeeze last channel to obtain shape (h,w,)

    if returnFigure:
        output["fig"] = plot_to_image(fig)

    if fileName:
        plt.savefig(os.path.join(fileName))
        plt.close(fig)
    else:
        plt.show()
    return output


def PlotHistogramm(img, title=None, returnFigure=False):
    """
    img is expected to be:
      - scaled to [-1,1]
      - of shape (1, h, w, channels)
    """
    output = {}
    fig = plt.figure()
    if len(img.shape) == 4:  # (batch, height, width, channels)
        img = np.squeeze(img, axis=0)  # squeeze batch dimension to obtain shape (h,w,channels)
    if title:
        plt.title(title)
    min = tf.reduce_min(img)
    max = tf.reduce_max(img)
    hist = tf.histogram_fixed_width(img, [min, max], nbins=512)
    x_Range = np.linspace(min, max, 512)
    plt.plot(x_Range, hist)
    plt.fill_between(x_Range, 0, hist)
    plt.yscale("log")
    plt.xticks([])
    plt.yticks([])

    if returnFigure:
        output["fig"] = plot_to_image(fig)
        plt.close(fig)
    return output


def PointCloud(
    depth,
    ax,
    img=None,
    step=1,
    elevation=0,
    azimuth=0,
    threshold=-1,
    hide_axis_and_grid=False,
    linewidths=0.2,
    drop_shaddow=False,
    correlate_depth_img=False,
    place_on_ground=False,
    marker=".",
):
    depth = depth[..., -1]
    if img is not None:
        img = (img + 1) / 2
        img_shape = np.shape(img)
        img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)
    depth_shape = np.shape(depth)
    ax.view_init(elevation, azimuth)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, depth_shape[0])
    ax.set_zlim(0, depth_shape[0])
    y_axis = range(0, depth_shape[0], step)
    z_axis = range(0, depth_shape[1], step)

    yy, zz = np.meshgrid(y_axis, z_axis)

    if correlate_depth_img and img is not None:
        threshold_depth = threshold
        threshold_img = 0
        # mask for RGB is created by summing over channels. ultimately i am interested in removing the background which is (0,0,0) in RGB. if after summation, the result is still zero, the pixel was black.
        # open cv's cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) did not work for float image [0...1]
        mask_img = np.sum(img, axis=-1)
        mask_img = tf.where(mask_img > threshold_img, 1, 0)
        mask_depth = tf.where(depth > threshold_depth, 1, 0)

        correlation = signal.correlate2d(mask_img, mask_depth, mode="same")
        # get position of max correlation. Interpreted as deviation from center.
        y, x = np.unravel_index(np.argmax(correlation), correlation.shape)

        depth = np.roll(depth, y - depth_shape[0] // 2, axis=0)
        depth = np.roll(depth, x - depth_shape[1] // 2, axis=1)

    if place_on_ground:
        # get 0/1 mask
        mask = tf.where(depth > threshold, 1, 0)
        # sum along width axis to obtain column vector. if an element is 0, it means no depth point in a given row
        reduced_mask = np.sum(mask, axis=1, keepdims=True)
        # get index of last non-zero element
        lowest_point = np.flatnonzero(reduced_mask)[-1]
        depth = np.roll(depth, depth_shape[0] - lowest_point - 1, axis=0)
        if img is not None:
            img = np.roll(img, depth_shape[0] - lowest_point - 1, axis=0)

    depth = np.flipud(depth)
    if img is not None:
        img = np.flipud(img)

    # threshold removes points with depth smaller the treshold and where the color of the pixel is black
    depth = np.where(depth <= threshold, np.nan, np.where(depth > 1, np.nan, depth))
    x = depth[yy, zz]
    y = zz
    z = yy
    if img is not None:
        color = np.swapaxes(img, 0, 1).reshape(img_shape[0] * img_shape[1], img_shape[2])
        cmap = None
    else:
        color = x[x != np.nan]
        cmap = "viridis"
    ax.scatter(x, y, z, marker=marker, c=color, cmap=cmap, linewidths=linewidths)

    if drop_shaddow:
        gauss = DropShadowFilter(10, alpha=0.6, offsets=(5, 5))
        shadow = ax.scatter(x, y, marker=".", c="black", linewidths=1)
        shadow.set_agg_filter(gauss)

    if hide_axis_and_grid:
        plt.axis("off")


def PlotPointCloud(
    depth,
    img=None,
    step=1,
    title=None,
    returnFigure=False,
    threshold=-1,
    fileName=None,
    hide_axis_and_grid=False,
    linewidths=0.2,
    elevations=[10, 10, 10, 90, 10, 10],
    azimuths=[-90, -45, 0, 0, 45, 90],
    drop_shaddow=False,
    fig_size=None,
    correlate_depth_img=False,
    place_on_ground=False,
    marker=".",
):
    """
    plots a 3D point cloud from a given RGBD or grayscale image
    RGBD [-1:1]

    """
    assert len(elevations) == len(azimuths)
    num_plots = len(elevations)
    output = {}

    fig = plt.figure()
    if fig_size is None:
        fig = plt.figure(figsize=(num_plots * 5, 5))
    else:
        fig = plt.figure(figsize=fig_size)
    if title is not None:
        fig.suptitle(title, fontsize=20)

    for plot, (elevation, azimuth) in enumerate(zip(elevations, azimuths)):
        ax = fig.add_subplot(1, num_plots, plot + 1, projection="3d")
        PointCloud(
            depth,
            ax,
            img=img,
            step=step,
            elevation=elevation,
            azimuth=azimuth,
            threshold=threshold,
            hide_axis_and_grid=hide_axis_and_grid,
            linewidths=linewidths,
            drop_shaddow=drop_shaddow,
            marker=marker,
            place_on_ground=place_on_ground,
            correlate_depth_img=correlate_depth_img,
        )

    if returnFigure:
        output["fig"] = plot_to_image(fig)

    if fileName:
        plt.savefig(os.path.join(fileName), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return output


def PlotDualPointCloud(
    depth1,
    depth2,
    img1=None,
    img2=None,
    step=1,
    title=None,
    returnFigure=False,
    threshold=-1,
    fileName=None,
    hide_axis_and_grid=False,
    linewidths=0.2,
    elevations=[5, 5, 5, 90, 5, 5],
    azimuths=[-90, -45, 0, 0, 45, 90],
    drop_shaddow=False,
    fig_size=None,
    correlate_depth_img=False,
    place_on_ground=False,
    marker=".",
):
    """plots a 3D point cloud from a given RGBD or grayscale image"""

    assert len(elevations) == len(azimuths)
    num_plots = len(elevations)
    output = {}

    if fig_size is None:
        fig = plt.figure(figsize=(num_plots * 5, 8))
    else:
        fig = plt.figure(figsize=fig_size)
    if title is not None:
        fig.suptitle(title, fontsize=20)

    for plot, (elevation, azimuth) in enumerate(zip(elevations, azimuths)):
        # Row 1 for input 1
        ax = fig.add_subplot(2, num_plots, plot + 1, projection="3d")
        PointCloud(
            depth1,
            ax,
            img=img1,
            step=step,
            elevation=elevation,
            azimuth=azimuth,
            threshold=threshold,
            hide_axis_and_grid=hide_axis_and_grid,
            linewidths=linewidths,
            drop_shaddow=drop_shaddow,
            marker=marker,
            place_on_ground=place_on_ground,
            correlate_depth_img=correlate_depth_img,
        )
        # Row 2 for input 21
        ax = fig.add_subplot(2, num_plots, plot + 1 + num_plots, projection="3d")
        PointCloud(
            depth2,
            ax,
            img=img2,
            step=step,
            elevation=elevation,
            azimuth=azimuth,
            threshold=threshold,
            hide_axis_and_grid=hide_axis_and_grid,
            linewidths=linewidths,
            drop_shaddow=drop_shaddow,
            marker=marker,
            place_on_ground=place_on_ground,
            correlate_depth_img=correlate_depth_img,
        )

    if returnFigure:
        output["fig"] = plot_to_image(fig)

    if fileName:
        plt.savefig(os.path.join(fileName), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return output


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    return tf.expand_dims(image, 0)


# TODO: timestep 0 is not plotted.
def PlotReverseDiffusion(x0_condition, xt_frames, timesteps, numberOfPlots=10, fileName=False):
    output = {}
    frames, height, width, channel = np.shape(xt_frames)
    numberOfPlots = min(numberOfPlots, frames)
    fig = plt.figure(figsize=(numberOfPlots * 3, 3))
    frames_to_plot = np.linspace(0, frames + 1, numberOfPlots + 1, dtype="int32")
    ax = fig.add_subplot(1, len(frames_to_plot) + 1, 1)
    PlotSample(x0_condition, ax, title="Condition")
    plot_counter = 2

    for i, (frame, t) in enumerate(zip(xt_frames, timesteps)):
        if i in frames_to_plot:
            ax = fig.add_subplot(1, numberOfPlots + 1, plot_counter)
            PlotSample(tf.expand_dims(frame, axis=0), ax, title=f"t = {t}")
            plot_counter += 1

    if fileName:
        plt.savefig(os.path.join(fileName), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return output


def PlotBatchedSample(
    x0_condition, x0_diffusion_input, samplerOutput, num_samples=-1, run_output_dir="", postfix="", epoch=None
):
    batch_size, frames, height, width, channels = samplerOutput["xt_frames"].shape
    samples_to_plot = batch_size if num_samples == -1 else min(batch_size, num_samples)

    for i, sample in enumerate(samplerOutput["xt_frames"][0:samples_to_plot, ...]):
        fileName_pc = os.path.join(run_output_dir, "illustrations", f"pointcloud_{postfix}_{i}.png")
        fileName_rd = os.path.join(run_output_dir, "illustrations", f"reverse_diffusion_{postfix}_{i}.png")
        PlotReverseDiffusion(
            x0_condition[i, ...], sample, samplerOutput["t_frames"], numberOfPlots=8, fileName=fileName_rd
        )
        PlotDualPointCloud(
            x0_diffusion_input[i, ...],
            samplerOutput["x0"][i, ...],
            img1=x0_condition[i, ...],
            img2=x0_condition[i, ...],
            title="Point Cloud. (Top) Real X0. (Bottom) Diffused X0",
            threshold=-0.9,
            hide_axis_and_grid=True,
            drop_shaddow=True,
            fileName=fileName_pc,
        )


def PlotBatchedSampleSuperRes(
    x0_condition,
    x0_diffusion_input,
    samplerOutput,
    condition_format,
    diffusion_format,
    num_samples=-1,
    run_output_dir="",
    postfix="",
    epoch=None,
):
    batch_size, height, width, channels = samplerOutput["x0"].shape
    samples_to_plot = batch_size if num_samples == -1 else min(batch_size, num_samples)

    for i, sample in enumerate(samplerOutput["x0"][0:samples_to_plot, ...]):
        if condition_format == "rgbd":
            x0_img_low_res, x0_depth_low_res = tf.split(x0_condition[i, ...], [3, 1], axis=-1)
        elif condition_format == "depth":
            x0_depth_low_res = x0_condition[i, ...]
            x0_img_low_res = None

        if diffusion_format == "rgbd":
            x0_img_high_res, x0_depth_high_res = tf.split(x0_diffusion_input[i, ...], [3, 1], axis=-1)
            x0_img_high_res_diffused, x0_depth_high_res_diffused = tf.split(sample, [3, 1], axis=-1)
        elif diffusion_format == "depth":
            x0_depth_high_res = x0_diffusion_input[i, ...]
            x0_img_high_res = None
            x0_depth_high_res_diffused = sample
            x0_img_high_res_diffused = None

        fileName_pc = os.path.join(run_output_dir, "illustrations", f"pointcloud_{postfix}_{i}.png")
        fileName_sr = os.path.join(run_output_dir, "illustrations", f"comparison_{postfix}_{i}.png")
        PlotDualPointCloud(
            x0_depth_low_res,
            x0_depth_high_res_diffused,
            img1=x0_img_low_res,
            img2=x0_img_high_res_diffused,
            title="Point Cloud. (Top) Low Res X0. (Bottom) Diffused High Res X0",
            threshold=-0.9,
            hide_axis_and_grid=True,
            drop_shaddow=True,
            fileName=fileName_pc,
        )
        PlotSuperresComparison(
            x0_depth_low_res,
            x0_depth_high_res,
            x0_depth_high_res_diffused,
            x0_img_low_res,
            x0_img_high_res,
            x0_img_high_res_diffused,
            fileName=fileName_sr,
        )


def PlotSuperresComparison(
    depth_low_res,
    depth_high_res_orig,
    depth_high_res_generated,
    rgb_low_res=None,
    rgb_high_res_orig=None,
    rgb_high_res_generated=None,
    threshold=-1,
    fileName=None,
):
    plot_rgb = rgb_low_res is not None and rgb_high_res_orig is not None and rgb_high_res_generated is not None
    shape = depth_high_res_orig.shape
    output = {}
    columns = 5
    rows = 2 if plot_rgb else 1
    fig = plt.figure(figsize=(20, 5))
    # Low Res
    ax = fig.add_subplot(rows, columns, 1)
    PlotSample(depth_low_res, ax, title="Low Res", threshold=threshold)
    if plot_rgb:
        ax = fig.add_subplot(rows, columns, columns + 1)
        PlotSample(rgb_low_res, ax, threshold=threshold)

    # Nearest Upsampled High Res
    depth_nearest = tf.image.resize(depth_low_res, (shape[0], shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ax = fig.add_subplot(rows, columns, 2)
    PlotSample(depth_nearest, ax, title="Nearest", threshold=threshold)
    if plot_rgb:
        rgb_nearest = tf.image.resize(rgb_low_res, (shape[0], shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        ax = fig.add_subplot(rows, columns, columns + 2)
        PlotSample(rgb_nearest, ax, threshold=threshold)

    # Biliniear Upsampled High Res
    depth_bilinear = tf.image.resize(depth_low_res, (shape[0], shape[1]), method=tf.image.ResizeMethod.BILINEAR)
    ax = fig.add_subplot(rows, columns, 3)
    PlotSample(depth_bilinear, ax, title="Biliniear", threshold=threshold)
    if plot_rgb:
        rgb_bilinear = tf.image.resize(rgb_low_res, (shape[0], shape[1]), method=tf.image.ResizeMethod.BILINEAR)
        ax = fig.add_subplot(rows, columns, columns + 3)
        PlotSample(rgb_bilinear, ax, threshold=threshold)

    # Diffused High Res
    ax = fig.add_subplot(rows, columns, 4)
    PlotSample(depth_high_res_generated, ax, title="Diffused (ours)", threshold=threshold)
    if plot_rgb:
        ax = fig.add_subplot(rows, columns, columns + 4)
        PlotSample(rgb_high_res_generated, ax, threshold=threshold)

    # Original High Res
    ax = fig.add_subplot(rows, columns, 5)
    PlotSample(depth_high_res_orig, ax, title="GT", threshold=threshold)
    if plot_rgb:
        ax = fig.add_subplot(rows, columns, columns + 5)
        PlotSample(rgb_high_res_orig, ax, threshold=threshold)

    if fileName:
        plt.savefig(os.path.join(fileName), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return output
