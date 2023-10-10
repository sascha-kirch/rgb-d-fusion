import logging
import os
import random

import numpy as np
import tensorflow as tf

import cv2
import tensorflow_addons as tfa


def get_preprocess_superres_func(
    condition_format,  # rgb, depth, rgbd
    diffusion_format,  # rgb, depth, rgbd
    dtype,
    low_res_height_width,
):
    """Function to return preprocessing function for superresolution dataset, that returns a high-res and a low-res data sample.

    Args:
        dtype (numpy.dtype): dtype to which the data should be casted to.
        low_res_height_width (list): (height, width) to which the low resolution image shall be resized to using nearest neighbor interpolation.

    Returns:
        func: Function that resizes the spatial width of RGBD data using nearest neighbor interpolation to return a high resolution version and a low resolution version of the input and casts to the dtype provided.
    """
    # This means if condition RGBD, load RGBD highres, and if diffusion is depth onl, only return depth for high res

    high_res_is_rgbd = condition_format != diffusion_format or condition_format == "rgbd" or diffusion_format == "rgbd"

    def preprocess_superres(high_res):
        if high_res_is_rgbd:
            high_res_renders, high_res_depths = tf.split(high_res, [3, 1], axis=-1)
        elif diffusion_format == "depth":
            high_res_depths = high_res
        elif diffusion_format == "rgb":
            high_res_renders = high_res

        if condition_format == "rgb":
            cond = tf.image.resize(
                tf.cast(high_res_renders, dtype), low_res_height_width, method=tf.image.ResizeMethod.BILINEAR
            )
        elif condition_format == "depth":
            cond = tf.image.resize(
                tf.cast(high_res_depths, dtype), low_res_height_width, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
        elif condition_format == "rgbd":
            low_res_renders = tf.image.resize(
                tf.cast(high_res_renders, dtype), low_res_height_width, method=tf.image.ResizeMethod.BILINEAR
            )
            low_res_depths = tf.image.resize(
                tf.cast(high_res_depths, dtype), low_res_height_width, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            cond = tf.concat([low_res_renders, low_res_depths], axis=-1)

        if diffusion_format == "rgb":
            diff = high_res_renders
        elif diffusion_format == "depth":
            diff = high_res_depths
        elif diffusion_format == "rgbd":
            diff = high_res

        diff = tf.cast(diff, dtype)

        return cond, diff

    return preprocess_superres


def scale_tensor(input_tensor, scaleFactorMin, scaleFactorMax, method, padding_value=0):
    tensor_shape = input_tensor.shape
    scaleFactor = np.random.uniform(low=scaleFactorMin, high=scaleFactorMax)
    heightReductionOneSided = int(
        (tensor_shape[1] * (1 - scaleFactor)) // 2
    )  # one sided reduction to ensure it is even for padding afer resize
    widthReductionOneSided = int(
        (tensor_shape[2] * (1 - scaleFactor)) // 2
    )  # respects aspect ratio -> works for cropped images as well
    newHeight, newWidth = int(tensor_shape[1] - 2 * heightReductionOneSided), int(
        tensor_shape[2] - 2 * widthReductionOneSided
    )
    output_tensor = tf.image.resize(input_tensor, (newHeight, newWidth), method=method)

    padding_tensor = [
        [0, 0],  # batch dim
        [heightReductionOneSided, heightReductionOneSided],  # height dim
        [widthReductionOneSided, widthReductionOneSided],  # width dim
        [0, 0],  # channel dim
    ]

    return tf.pad(output_tensor, padding_tensor, constant_values=padding_value)


def shift_tensor(input_tensor, verticalShiftMax, horicontalShiftMax, padding_value=0):
    tensor_shape = input_tensor.shape
    heightPaddingOneSided = int((tensor_shape[1] * verticalShiftMax) // 2)  # one sided padding
    widthPaddingOneSided = int(
        (tensor_shape[2] * horicontalShiftMax) // 2
    )  # respects aspect ratio -> works for cropped images as well
    padding_tensor = [
        [0, 0],  # batch dim
        [heightPaddingOneSided, heightPaddingOneSided],  # height dim
        [widthPaddingOneSided, widthPaddingOneSided],  # width dim
        [0, 0],  # channel dim
    ]

    output_tensor = tf.pad(input_tensor, padding_tensor, constant_values=padding_value)

    return tf.image.random_crop(value=output_tensor, size=tensor_shape, seed=10)


def get_augment_func(
    condition_format,  # rgb, depth, rgbd
    diffusion_format,  # rgb, depth, rgbd
    apply_flip=True,
    apply_scale=True,
    apply_shift=True,
    apply_rgb_blur=True,
    apply_depth_blur=True,
    flipProbability=0.5,
    scaleProbability=1.0,
    shiftProbability=1.0,
    blurProbability=0.5,
    scaleFactorMax=1.0,
    scaleFactorMin=0.8,
    horicontalShiftMax=0.2,
    verticalShiftMax=0.1,
    dtype=np.dtype("float32"),
    depth_mask_threshold=-0.8,
):
    @tf.function
    def augment(cond, diff):
        # flip left-right
        if apply_flip and np.random.binomial(n=1, p=flipProbability):
            cond = tf.image.flip_left_right(cond)
            diff = tf.image.flip_left_right(diff)

        # scale
        if apply_scale and np.random.binomial(n=1, p=scaleProbability):
            if condition_format == "rgb":
                cond = scale_tensor(
                    cond, scaleFactorMin, scaleFactorMax, padding_value=0, method=tf.image.ResizeMethod.BILINEAR
                )
            elif condition_format == "depth":
                cond = scale_tensor(
                    cond,
                    scaleFactorMin,
                    scaleFactorMax,
                    padding_value=-1,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                )
            elif condition_format == "rgbd":
                cond_renders, cond_depths = tf.split(cond, [3, 1], axis=-1)
                cond_renders = scale_tensor(
                    cond_renders, scaleFactorMin, scaleFactorMax, padding_value=0, method=tf.image.ResizeMethod.BILINEAR
                )
                cond_depths = scale_tensor(
                    cond_depths,
                    scaleFactorMin,
                    scaleFactorMax,
                    padding_value=-1,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                )
                cond = tf.concat([cond_renders, cond_depths], axis=-1)

            if diffusion_format == "rgb":
                diff = scale_tensor(
                    diff, scaleFactorMin, scaleFactorMax, padding_value=0, method=tf.image.ResizeMethod.BILINEAR
                )
            elif diffusion_format == "depth":
                diff = scale_tensor(
                    diff,
                    scaleFactorMin,
                    scaleFactorMax,
                    padding_value=-1,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                )
            elif diffusion_format == "rgbd":
                diff_renders, diff_depths = tf.split(diff, [3, 1], axis=-1)
                diff_renders = scale_tensor(
                    diff_renders, scaleFactorMin, scaleFactorMax, padding_value=0, method=tf.image.ResizeMethod.BILINEAR
                )
                diff_depths = scale_tensor(
                    diff_depths,
                    scaleFactorMin,
                    scaleFactorMax,
                    padding_value=-1,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                )
                diff = tf.concat([diff_renders, diff_depths], axis=-1)

        # shift
        if apply_shift and np.random.binomial(n=1, p=shiftProbability):
            if condition_format == "rgb":
                cond = shift_tensor(cond, verticalShiftMax, horicontalShiftMax, padding_value=0)
            elif condition_format == "depth":
                cond = shift_tensor(cond, verticalShiftMax, horicontalShiftMax, padding_value=-1)
            elif condition_format == "rgbd":
                cond_renders, cond_depths = tf.split(cond, [3, 1], axis=-1)
                cond_renders = shift_tensor(cond_renders, verticalShiftMax, horicontalShiftMax, padding_value=0)
                cond_depths = shift_tensor(cond_depths, verticalShiftMax, horicontalShiftMax, padding_value=-1)
                cond = tf.concat([cond_renders, cond_depths], axis=-1)

            if diffusion_format == "rgb":
                diff = shift_tensor(diff, verticalShiftMax, horicontalShiftMax, padding_value=0)
            elif diffusion_format == "depth":
                diff = shift_tensor(diff, verticalShiftMax, horicontalShiftMax, padding_value=-1)
            elif diffusion_format == "rgbd":
                diff_renders, diff_depths = tf.split(diff, [3, 1], axis=-1)
                diff_renders = shift_tensor(diff_renders, verticalShiftMax, horicontalShiftMax, padding_value=0)
                diff_depths = shift_tensor(diff_depths, verticalShiftMax, horicontalShiftMax, padding_value=-1)
                diff = tf.concat([diff_renders, diff_depths], axis=-1)

        # blur - only applied on condition
        if apply_rgb_blur and condition_format == "rgb":
            if np.random.binomial(n=1, p=blurProbability):
                cond = tfa.image.gaussian_filter2d(
                    cond, filter_shape=(3, 3), sigma=np.random.uniform(low=0.0, high=0.6)
                )
        elif apply_depth_blur and condition_format == "depth":
            if np.random.binomial(n=1, p=blurProbability):
                noise = tf.random.normal(
                    tf.shape(cond), mean=0.0, stddev=np.random.uniform(low=0.0, high=0.06)
                )  # stddev has been selected emprically
                depth_mask = tf.where(
                    cond > depth_mask_threshold, 1.0, 0.0
                )  # only add noise to the subject, not the background!!!
                noise *= depth_mask
                cond += noise
        elif condition_format == "rgbd" and (apply_rgb_blur or apply_depth_blur):
            cond_renders, cond_depths = tf.split(cond, [3, 1], axis=-1)
            if np.random.binomial(n=1, p=blurProbability):
                cond_renders = tfa.image.gaussian_filter2d(
                    cond_renders, filter_shape=(3, 3), sigma=np.random.uniform(low=0.0, high=0.6)
                )
            if np.random.binomial(n=1, p=blurProbability):
                noise_depth = tf.random.normal(
                    tf.shape(cond_depths), mean=0.0, stddev=np.random.uniform(low=0.0, high=0.06)
                )  # stddev has been selected emprically
                depth_mask = tf.where(
                    cond_depths > depth_mask_threshold, 1.0, 0.0
                )  # only add noise to the subject, not the background!!!
                noise_depth *= depth_mask
                cond_depths += noise_depth
            cond = tf.concat([cond_renders, cond_depths], axis=-1)

        return tf.cast(cond, dtype=dtype), tf.cast(diff, dtype=dtype)

    return augment


def _read_render(file, cropWidthHalf, img_height_width, dtype):
    render = cv2.imread(file)[:, :, ::-1]
    if cropWidthHalf:
        height, width, _ = np.shape(render)
        width_quarter = int(width / 4)
        render = render[0:height, width_quarter : width - width_quarter]
    if img_height_width:
        # CAUTION: resize requires outputdims in (widht,height), NOT in (height,width)
        render = cv2.resize(render, (img_height_width[1], img_height_width[0]), interpolation=cv2.INTER_LINEAR)
    return np.divide(render, 127.5, dtype=dtype) - 1


def _read_depth(file, cropWidthHalf, img_height_width, dtype, format="exr"):
    if format == "exr":
        depth = cv2.imread(file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    elif format == "png":
        depth = cv2.imread(file, 0)
    else:
        raise NotImplementedError
    if cropWidthHalf:
        height, width, _ = np.shape(depth)
        width_quarter = int(width / 4)
        depth = depth[0:height, width_quarter : width - width_quarter]
    if img_height_width:
        # CAUTION: resize requires outputdims in (widht,height), NOT in (height,width)
        depth = cv2.resize(depth, (img_height_width[1], img_height_width[0]), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(depth, axis=2).astype(dtype)


def prepare_streamed_dataset(
    readRgb,
    readDepth,
    datasetDirectory,
    img_height_width,
    NumberOfSamplesToRead,
    dtype,
    returnRgbd,
    cropWidthHalf,
    shuffle,
):
    assert readRgb and readDepth if returnRgbd else True

    def list_full_paths(directory):
        return [os.path.join(directory, file) for file in os.listdir(directory)]

    numberOfSamples = 0

    if readRgb:
        render_path = os.path.join(datasetDirectory, "RENDER")
        render_filenames = sorted(list_full_paths(render_path))
        numberOfSamples = np.shape(render_filenames)[0]
    else:
        render_filenames = None

    if readDepth:
        depth_path = os.path.join(datasetDirectory, "DEPTH_RENDER_EXR")
        depth_filenames = sorted(list_full_paths(depth_path))
        if numberOfSamples == 0:
            numberOfSamples = np.shape(depth_filenames)[0]
        else:
            assert numberOfSamples == np.shape(depth_filenames)[0]  # ensure depth and rgb have same number of samples
    else:
        depth_filenames = None

    if NumberOfSamplesToRead:
        # in case more samples are requested as there are available
        readCount = min(numberOfSamples, NumberOfSamplesToRead)
    else:
        readCount = numberOfSamples

    logging.info(f"{readCount} of {numberOfSamples} samples will be streamed.")

    def _get_generator():
        if readRgb and readDepth:
            files = list(zip(render_filenames, depth_filenames))[0:readCount]
            # initial shuffle, that is applied to train and test.
            random.shuffle(files)

            def _load_rgb_depth_generator():
                # randomize Filenames every epoch to randomize batches!
                if shuffle:
                    random.shuffle(files)
                for render_file, depth_file in files:
                    render = _read_render(render_file, cropWidthHalf, img_height_width, dtype)
                    depth = _read_depth(depth_file, cropWidthHalf, img_height_width, dtype)
                    if returnRgbd and readRgb and readDepth:
                        output = tf.concat([render, depth], axis=-1)
                    elif readRgb and readDepth:
                        output = (render, depth)
                    yield output

            return _load_rgb_depth_generator
        elif readRgb and not readDepth:
            files = render_filenames[0:readCount]
            # initial shuffle, that is applied to train and test.
            random.shuffle(files)

            def _load_rgb_generator():
                # randomize Filenames every epoch to randomize batches!
                if shuffle:
                    random.shuffle(files)
                for render_file in files:
                    yield _read_render(render_file, cropWidthHalf, img_height_width, dtype)

            return _load_rgb_generator
        elif readDepth and not readRgb:
            files = depth_filenames[0:readCount]
            # initial shuffle, that is applied to train and test.
            random.shuffle(files)

            def _load_depth_generator():
                # randomize Filenames every epoch to randomize batches!
                if shuffle:
                    random.shuffle(files)
                for depth_file in files:
                    yield _read_depth(depth_file, cropWidthHalf, img_height_width, dtype)

            return _load_depth_generator
        else:
            raise Exception

    if returnRgbd and readRgb and readDepth:
        output_signature = tf.TensorSpec(shape=(*img_height_width, 4), dtype=dtype)
    elif readRgb and readDepth:
        output_signature = (
            tf.TensorSpec(shape=(*img_height_width, 3), dtype=dtype),
            tf.TensorSpec(shape=(*img_height_width, 1), dtype=dtype),
        )
    elif readRgb and not readDepth:
        output_signature = tf.TensorSpec(shape=(*img_height_width, 3), dtype=dtype)
    elif not readRgb and readDepth:
        output_signature = tf.TensorSpec(shape=(*img_height_width, 1), dtype=dtype)

    # streaming data
    return tf.data.Dataset.from_generator(_get_generator(), output_signature=output_signature)


def GetDatasetDepthDiffusionStreamed(
    datasetDirectory,
    batchSize,
    img_height_width,
    NumberOfSamplesToRead=None,
    dtype=np.dtype("float32"),
    drop_remainder=False,
    cropWidthHalf=False,
    shuffle=True,
    apply_flip=False,
    apply_scale=False,
    apply_shift=False,
    apply_rgb_blur=False,
    apply_depth_blur=False,
):
    ds = prepare_streamed_dataset(
        True, True, datasetDirectory, img_height_width, NumberOfSamplesToRead, dtype, False, cropWidthHalf, shuffle
    )
    ds = ds.batch(batchSize, drop_remainder=drop_remainder)
    if any([apply_flip, apply_scale, apply_shift, apply_rgb_blur, apply_depth_blur]):
        ds = ds.map(
            get_augment_func(
                "rgb", "depth", apply_flip, apply_scale, apply_shift, apply_rgb_blur, apply_depth_blur, dtype=dtype
            ),
            tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def GetDatasetSuperresStreamed(
    datasetDirectory,
    batchSize,
    low_res_height_width,
    high_res_height_width,
    condition_format,
    diffusion_format,
    NumberOfSamplesToRead=None,
    dtype=np.dtype("float32"),
    drop_remainder=False,
    cropWidthHalf=False,
    shuffle=True,
    apply_flip=False,
    apply_scale=False,
    apply_shift=False,
    apply_rgb_blur=False,
    apply_depth_blur=False,
):
    read_rgbd = condition_format != diffusion_format or condition_format == "rgbd" or diffusion_format == "rgbd"

    # ds = prepare_streamed_dataset(condition_format in ["rgb","rgbd"], condition_format in ["depth","rgbd"], datasetDirectory,high_res_height_width,NumberOfSamplesToRead,dtype, condition_format =="rgbd",cropWidthHalf,shuffle)
    ds = prepare_streamed_dataset(
        condition_format in ["rgb"] or read_rgbd,
        condition_format in ["depth"] or read_rgbd,
        datasetDirectory,
        high_res_height_width,
        NumberOfSamplesToRead,
        dtype,
        read_rgbd,
        cropWidthHalf,
        shuffle,
    )
    ds = ds.map(
        get_preprocess_superres_func(condition_format, diffusion_format, dtype, low_res_height_width), tf.data.AUTOTUNE
    )
    ds = ds.batch(batchSize, drop_remainder=drop_remainder)

    if any([apply_flip, apply_scale, apply_shift, apply_rgb_blur, apply_depth_blur]):
        ds = ds.map(
            get_augment_func(
                condition_format,
                diffusion_format,
                apply_flip,
                apply_scale,
                apply_shift,
                apply_rgb_blur,
                apply_depth_blur,
            ),
            tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def GetDatasetDepthDiffusionStreamedForSampling(
    datasetDirectory,
    batchSize,
    img_height_width,
    NumberOfSamplesToRead=None,
    dtype=np.dtype("float32"),
    cropWidthHalf=False,
):
    ds = prepare_streamed_dataset(
        True,
        False,
        datasetDirectory,
        img_height_width,
        NumberOfSamplesToRead,
        dtype,
        False,
        cropWidthHalf,
        shuffle=False,
    )
    ds = ds.batch(batchSize, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)


def GetDatasetSuperresStreamedForSampling(
    datasetDirectory,
    batchSize,
    low_res_height_width,
    condition_format,
    NumberOfSamplesToRead=None,
    dtype=np.dtype("float32"),
    cropWidthHalf=False,
):
    ds = prepare_streamed_dataset(
        condition_format in ["rgb", "rgbd"],
        condition_format in ["depth", "rgbd"],
        datasetDirectory,
        low_res_height_width,
        NumberOfSamplesToRead,
        dtype,
        condition_format == "rgbd",
        cropWidthHalf,
        shuffle=False,
    )
    ds = ds.batch(batchSize, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)
