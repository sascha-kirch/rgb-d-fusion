import logging
import os
import random
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def get_preprocess_superres_func(
    condition_format: str,
    diffusion_format: str,
    dtype: Union[tf.DType, np.dtype],
    low_res_height_width: Tuple[int, int],
) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """Get preprocessing function for the super resolution model.

    Args:
        condition_format (str): Data format of the condition input: [ rgb | depth | rgbd ]
        diffusion_format (str): Data format of the diffusion input: [ rgb | depth | rgbd ]
        dtype (Union[tf.DType, np.dtype]): dtype to which the data should be casted to.
        low_res_height_width (Tuple[int, int]): (height, width) to which the low resolution image shall be resized to
            using nearest neighbor interpolation.

    Returns:
        Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]: Function that resizes the spatial width of RGBD data using
            nearest neighbor interpolation to return a high resolution version and a low resolution version of the
            input and casts to the dtype provided.
    """
    # This means if condition RGBD, load RGBD highres, and if diffusion is depth onl, only return depth for high res

    high_res_is_rgbd = condition_format != diffusion_format or condition_format == "rgbd" or diffusion_format == "rgbd"

    def _preprocess_superres(high_res: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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

    return _preprocess_superres


def scale_tensor(
    input_tensor: tf.Tensor,
    scaleFactorMin: float,
    scaleFactorMax: float,
    method: tf.image.ResizeMethod,
    padding_value: int = 0,
) -> tf.Tensor:
    """Randomly scales a person depicted in the input tensor regardles of the format beeing `depth`, `rgb` or `rgbd`.

    Args:
        input_tensor (tf.Tensor): Input tensor to be scaled.
        scaleFactorMin (float): Minimum scaling factor
        scaleFactorMax (float): Maximum scaling factor
        method (tf.image.ResizeMethod): Method to resize / resample the tensor.
        padding_value (int, optional): Value used for padding the scaled tensor. Defaults to 0.

    Returns:
        tf.Tensor: Tensor containing a scaled version of the person depicted in the tensor image.
    """
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


def shift_tensor(
    input_tensor: tf.Tensor,
    verticalShiftMax: float,
    horicontalShiftMax: float,
    padding_value: int = 0,
) -> tf.Tensor:
    """Randomly shifts a person depicted in the input tensor regardles of the format beeing `depth`, `rgb` or `rgbd`.

    Args:
        input_tensor (tf.Tensor): Input tensor to be shifted.
        verticalShiftMax (float): Maximum vertical shift.
        horicontalShiftMax (float): Maximum horicontal shift.
        padding_value (int, optional): Value used for padding the shifted tensor. Defaults to 0.

    Returns:
        tf.Tensor: Tensor containing a shifted version of the person depicted in the tensor image.
    """
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
    condition_format: str,
    diffusion_format: str,
    apply_flip: bool = True,
    apply_scale: bool = True,
    apply_shift: bool = True,
    apply_rgb_blur: bool = True,
    apply_depth_blur: bool = True,
    flipProbability: float = 0.5,
    scaleProbability: float = 1.0,
    shiftProbability: float = 1.0,
    blurProbability: float = 0.5,
    scaleFactorMax: float = 1.0,
    scaleFactorMin: float = 0.8,
    horicontalShiftMax: float = 0.2,
    verticalShiftMax: float = 0.1,
    dtype: Union[tf.DType, np.dtype] = np.dtype("float32"),
    depth_mask_threshold: float = -0.8,
) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """Get function implementing various augmentations.

    Args:
        condition_format (str): Data format of the condition input: [ rgb | depth | rgbd ]
        diffusion_format (str): Data format of the diffusion input: [ rgb | depth | rgbd ]
        apply_flip (bool, optional): If true, randomly flip tensor. Defaults to True.
        apply_scale (bool, optional): If true, randomly scale tensor. Defaults to True.
        apply_shift (bool, optional): If true, randomly shift tensor. Defaults to True.
        apply_rgb_blur (bool, optional): If true, randomly blur image. Defaults to True.
        apply_depth_blur (bool, optional): If true, randomly blur depth. Defaults to True.
        flipProbability (float, optional): Probability to flip tensor. Defaults to 0.5.
        scaleProbability (float, optional): Probability to scale tensor. Defaults to 1.0.
        shiftProbability (float, optional): Probability to shift tensor. Defaults to 1.0.
        blurProbability (float, optional): Probability to blur image. Defaults to 0.5.
        scaleFactorMax (float, optional): Maximum scaling factor to scale tensor. Defaults to 1.0.
        scaleFactorMin (float, optional): Minimum scaling factor to scale tensor. Defaults to 0.8.
        horicontalShiftMax (float, optional): Maximum shifting factor for horizontal shift. Defaults to 0.2.
        verticalShiftMax (float, optional): Maximum shifting factor for vertical shift. Defaults to 0.1.
        dtype (Union[tf.DType, np.dtype], optional): dtype to which the data should be casted to.
            Defaults to np.dtype("float32").
        depth_mask_threshold (float, optional): Threshold for removing background pixel. Defaults to -0.8.

    Returns:
        Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]: Augmentation function.
    """

    @tf.function
    def augment(cond: tf.Tensor, diff: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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


def _read_render(
    file: str,
    cropWidthHalf: bool,
    img_height_width: Tuple[int, int],
    dtype: Union[tf.DType, np.dtype],
) -> np.ndarray:
    """Read render file aka. RGB.

    Args:
        file (str): path to file.
        cropWidthHalf (bool): If true, width is center cropped into half the width.
        img_height_width (Tuple[int, int]): Resize image to (height, width)
        dtype (Union[tf.DType, np.dtype]): dtype to which the data should be casted to.

    Returns:
        np.ndarray: RGB image.
    """
    render = cv2.imread(file)[:, :, ::-1]
    if cropWidthHalf:
        height, width, _ = np.shape(render)
        width_quarter = int(width / 4)
        render = render[0:height, width_quarter : width - width_quarter]
    if img_height_width:
        # CAUTION: resize requires outputdims in (widht,height), NOT in (height,width)
        render = cv2.resize(render, (img_height_width[1], img_height_width[0]), interpolation=cv2.INTER_LINEAR)
    return np.divide(render, 127.5, dtype=dtype) - 1


def _read_depth(
    file: str,
    cropWidthHalf: bool,
    img_height_width: Tuple[int, int],
    dtype: Union[tf.DType, np.dtype],
    format: str = "exr",
) -> np.ndarray:
    """Read depth map file.

    Args:
        file (str): path to file.
        cropWidthHalf (bool): If true, width is center cropped into half the width.
        img_height_width (Tuple[int, int]): Resize image to (height, width)
        dtype (Union[tf.DType, np.dtype]): dtype to which the data should be casted to.
        format (str, optional): File format: [ exr | png ]. Defaults to "exr".

    Raises:
        NotImplementedError: _description_

    Returns:
        np.ndarray: Depth map.
    """
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
    readRgb: bool,
    readDepth: bool,
    datasetDirectory: str,
    img_height_width: Tuple[int, int],
    NumberOfSamplesToRead: Optional[int],
    dtype: Union[tf.DType, np.dtype],
    returnRgbd: bool,
    cropWidthHalf: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    """Get streamed dataset yielded from a generator.

    Args:
        readRgb (bool): If true, read RGB data.
        readDepth (bool): If True, read depth map.
        datasetDirectory (str): Path to directory containing dataset.
        img_height_width (Tuple[int, int]): Resize data to (height, width)
        NumberOfSamplesToRead (Optional[int]): Number of data samples to read from the datasset.
        dtype (Union[tf.DType, np.dtype]): dtype to which the data should be casted to.
        returnRgbd (bool): If true, RGB image and depth map are concatenated into single data structure.
        cropWidthHalf (bool): If true, width is center cropped into half the width.
        shuffle (bool): If true, dataset is shuffled.

    Returns:
        tf.data.Dataset: Dataset.

    """
    assert readRgb and readDepth if returnRgbd else True

    def _list_full_paths(directory: str) -> List[str]:
        return [os.path.join(directory, file) for file in os.listdir(directory)]

    numberOfSamples: int = 0
    render_filenames: List[str] = []
    depth_filenames: List[str] = []

    if readRgb:
        render_path = os.path.join(datasetDirectory, "RENDER")
        render_filenames = sorted(_list_full_paths(render_path))
        numberOfSamples = np.shape(render_filenames)[0]

    if readDepth:
        depth_path = os.path.join(datasetDirectory, "DEPTH_RENDER_EXR")
        depth_filenames = sorted(_list_full_paths(depth_path))
        if numberOfSamples == 0:
            numberOfSamples = np.shape(depth_filenames)[0]
        else:
            assert numberOfSamples == np.shape(depth_filenames)[0]  # ensure depth and rgb have same number of samples

    readCount = min(numberOfSamples, NumberOfSamplesToRead) if NumberOfSamplesToRead is not None else numberOfSamples

    logging.info(f"{readCount} of {numberOfSamples} samples will be streamed.")

    def _get_generator() -> Callable[[], Generator]:
        if readRgb and readDepth:
            files_combined = list(zip(render_filenames, depth_filenames))[0:readCount]
            # initial shuffle, that is applied to train and test.
            random.shuffle(files_combined)

            def _load_rgb_depth_generator() -> Generator[Union[tf.Tensor, Tuple[np.ndarray, np.ndarray]], None, None]:
                # randomize Filenames every epoch to randomize batches!
                if shuffle:
                    random.shuffle(files_combined)
                for render_file, depth_file in files_combined:
                    render = _read_render(render_file, cropWidthHalf, img_height_width, dtype)
                    depth = _read_depth(depth_file, cropWidthHalf, img_height_width, dtype)
                    if returnRgbd and readRgb and readDepth:
                        output = tf.concat([render, depth], axis=-1)
                    elif readRgb and readDepth:
                        output = (render, depth)
                    yield output

            return _load_rgb_depth_generator
        elif readRgb and not readDepth:
            files_render = render_filenames[0:readCount]
            # initial shuffle, that is applied to train and test.
            random.shuffle(files_render)

            def _load_rgb_generator() -> Generator[np.ndarray, None, None]:
                # randomize Filenames every epoch to randomize batches!
                if shuffle:
                    random.shuffle(files_render)
                for render_file in files_render:
                    yield _read_render(render_file, cropWidthHalf, img_height_width, dtype)

            return _load_rgb_generator
        elif readDepth and not readRgb:
            files_depth = depth_filenames[0:readCount]
            # initial shuffle, that is applied to train and test.
            random.shuffle(files_depth)

            def _load_depth_generator() -> Generator[np.ndarray, None, None]:
                # randomize Filenames every epoch to randomize batches!
                if shuffle:
                    random.shuffle(files_depth)
                for depth_file in files_depth:
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
    datasetDirectory: str,
    batchSize: int,
    img_height_width: Tuple[int, int],
    NumberOfSamplesToRead: Optional[int] = None,
    dtype: Union[tf.DType, np.dtype] = np.dtype("float32"),
    drop_remainder: bool = False,
    cropWidthHalf: bool = False,
    shuffle: bool = True,
    apply_flip: bool = False,
    apply_scale: bool = False,
    apply_shift: bool = False,
    apply_rgb_blur: bool = False,
    apply_depth_blur: bool = False,
) -> tf.data.Dataset:
    """Get streamed dataset to train/test the depth diffusion model.

    Args:
        datasetDirectory (str): Path to directory containing dataset.
        batchSize (int): Desired batch size of the dataset.
        img_height_width (Tuple[int, int]): Resize data to (height, width)
        NumberOfSamplesToRead (Optional[int], optional): Number of data samples to read from the datasset.
            Defaults to None.
        dtype (Union[tf.DType, np.dtype], optional): dtype to which the data should be casted to.
            Defaults to np.dtype("float32").
        drop_remainder (bool, optional): If true, remaining samples that do not fill a whole batch are dropped.
            Defaults to False.
        cropWidthHalf (bool, optional): If true, width is center cropped into half the width. Defaults to False.
        shuffle (bool, optional): If true, dataset is shuffled. Defaults to True.
        apply_flip (bool, optional): If true, randomly flip data sample. Defaults to False.
        apply_scale (bool, optional): If true, randomly scale data sample. Defaults to False.
        apply_shift (bool, optional): If true, randomly shift data sample. Defaults to False.
        apply_rgb_blur (bool, optional): If true, randomly blur RGB sample. Defaults to False.
        apply_depth_blur (bool, optional): If true, randomly blur depth sample. Defaults to False.

    Returns:
        tf.data.Dataset: Dataset.
    """
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
    datasetDirectory: str,
    batchSize: int,
    low_res_height_width: Tuple[int, int],
    high_res_height_width: Tuple[int, int],
    condition_format: str,
    diffusion_format: str,
    NumberOfSamplesToRead: Optional[int] = None,
    dtype: Union[tf.DType, np.dtype] = np.dtype("float32"),
    drop_remainder: bool = False,
    cropWidthHalf: bool = False,
    shuffle: bool = True,
    apply_flip: bool = False,
    apply_scale: bool = False,
    apply_shift: bool = False,
    apply_rgb_blur: bool = False,
    apply_depth_blur: bool = False,
) -> tf.data.Dataset:
    """Get streamed dataset to train/test the depth super resolution diffusion model.

    Args:
        datasetDirectory (str): Path to directory containing dataset.
        batchSize (int): Desired batch size of the dataset.
        low_res_height_width (Tuple[int, int]): Resize low resolution input data to (height, width)
        high_res_height_width (Tuple[int, int]): Resize high resolution input data to (height, width)
        condition_format (str): Data format of the condition input: [ rgb | depth | rgbd ]
        diffusion_format (str): Data format of the diffusion input: [ rgb | depth | rgbd ]
        NumberOfSamplesToRead (Optional[int], optional): Number of data samples to read from the datasset.
            Defaults to None.
        dtype (Union[tf.DType, np.dtype], optional): dtype to which the data should be casted to.
            Defaults to np.dtype("float32").
        drop_remainder (bool, optional): If true, remaining samples that do not fill a whole batch are dropped.
            Defaults to False.
        cropWidthHalf (bool, optional): If true, width is center cropped into half the width. Defaults to False.
        shuffle (bool, optional): If true, dataset is shuffled. Defaults to True.
        apply_flip (bool, optional): If true, randomly flip data sample. Defaults to False.
        apply_scale (bool, optional): If true, randomly scale data sample. Defaults to False.
        apply_shift (bool, optional): If true, randomly shift data sample. Defaults to False.
        apply_rgb_blur (bool, optional): If true, randomly blur RGB sample. Defaults to False.
        apply_depth_blur (bool, optional): If true, randomly blur depth sample. Defaults to False.

    Returns:
        tf.data.Dataset: Dataset.
    """
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
    datasetDirectory: str,
    batchSize: int,
    img_height_width: Tuple[int, int],
    NumberOfSamplesToRead: Optional[int] = None,
    dtype: Union[tf.DType, np.dtype] = np.dtype("float32"),
    cropWidthHalf: bool = False,
) -> tf.data.Dataset:
    """Get streamed dataset to sample from the depth diffusion model.

    Args:
        datasetDirectory (str): Path to directory containing dataset.
        batchSize (int): Desired batch size of the dataset.
        img_height_width (Tuple[int, int]): Resize data to (height, width)
        NumberOfSamplesToRead (Optional[int], optional): Number of data samples to read from the datasset.
            Defaults to None.
        dtype (Union[tf.DType, np.dtype], optional): dtype to which the data should be casted to.
            Defaults to np.dtype("float32").
        cropWidthHalf (bool, optional): If true, width is center cropped into half the width. Defaults to False.

    Returns:
        tf.data.Dataset: Dataset
    """
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
    datasetDirectory: str,
    batchSize: int,
    low_res_height_width: Tuple[int, int],
    condition_format: str,
    NumberOfSamplesToRead: Optional[int] = None,
    dtype: Union[tf.DType, np.dtype] = np.dtype("float32"),
    cropWidthHalf: bool = False,
) -> tf.data.Dataset:
    """Get streamed dataset to sample from the depth super resolution diffusion model.

    Args:
        datasetDirectory (str): Path to directory containing dataset.
        batchSize (int): Desired batch size of the dataset.
        low_res_height_width (Tuple[int, int]): Resize low resolution input data to (height, width)
        condition_format (str): Data format of the condition input: [ rgb | depth | rgbd ]
        NumberOfSamplesToRead (Optional[int], optional): Number of data samples to read from the datasset.
            Defaults to None.
        dtype (Union[tf.DType, np.dtype], optional): dtype to which the data should be casted to.
            Defaults to np.dtype("float32").
        cropWidthHalf (bool, optional): If true, width is center cropped into half the width. Defaults to False.

    Returns:
        tf.data.Dataset: Dataset.
    """
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
