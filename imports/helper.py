import os
import sys

# Get the parent directory to be able to import the files located in imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import yaml
from scipy import signal

import imports.callbacks as cdd_callbacks
import imports.model as cdd_model
import imports.plotting as cdd_plotting


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_channels_from_format(format: str) -> int:
    """Get number of channels required for a given `format`.

    Args:
        format (str): data format: [ rgb | depth | rgbd ]

    Raises:
        AttributeError: if `format` is not supported

    Returns:
        int: number of channels.
    """
    if format == "rgb":
        return 3
    if format == "depth":
        return 1
    if format == "rgbd":
        return 4
    raise AttributeError(f"Format '{format}' is not supported.")


def SaveDiffusedSamples(
    batchedSamples: tf.Tensor,
    outdir: str,
    dataFormat: str,
    fileNamePrefix: str = "",
) -> None:
    """Save data in appropiate file format.

    Args:
        batchedSamples (tf.Tensor): samples to be saved. Expected shape (batch, height, width, channel)
        outdir (str): Directory where to save the output.
        dataFormat (str): foramt of the data to be saved: [ rgb | depth | rgbd ]
        fileNamePrefix (str, optional): Prefix for filename to differentiate different outputs. Defaults to "".

    Raises:
        ValueError: if `dataformat` is not supported.
    """
    logging.info(f"Saving {dataFormat} outputs to: {outdir}")
    if dataFormat == "rgb":
        file_path = f"{outdir}/RENDER"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        for i, sample in enumerate(batchedSamples):
            image = np.array(sample[:, :, 0:3])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # change color format before saving
            image = np.array((image + 1) * 127.5, np.uint8)
            with tf.device("/job:localhost"):
                cv2.imwrite(f"{file_path}/{fileNamePrefix}_{i}.png", image)
    elif dataFormat == "depth":
        file_path = f"{outdir}/DEPTH_RENDER_EXR"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        for i, sample in enumerate(batchedSamples):
            depth = np.array(sample[:, :, 0:1])
            with tf.device("/job:localhost"):
                cv2.imwrite(f"{file_path}/{fileNamePrefix}_{i}.exr", depth, [cv2.IMWRITE_EXR_TYPE_FLOAT])
    elif dataFormat == "rgbd":
        render_path = f"{outdir}/RENDER"
        depth_path = f"{outdir}/DEPTH_RENDER_EXR"
        if not os.path.exists(render_path):
            os.mkdir(render_path)
        if not os.path.exists(depth_path):
            os.mkdir(depth_path)
        for i, sample in enumerate(batchedSamples):
            image = np.array(sample[:, :, 0:3])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # change color format before saving
            image = np.array((image + 1) * 127.5, np.uint8)
            depth = np.array(sample[:, :, 3:4])
            with tf.device("/job:localhost"):
                cv2.imwrite(f"{render_path}/{fileNamePrefix}_{i}.png", image)
                cv2.imwrite(f"{depth_path}/{fileNamePrefix}_{i}.exr", depth, [cv2.IMWRITE_EXR_TYPE_FLOAT])
    else:
        raise ValueError(f"Dataformat '{dataFormat}' not supported.")

    """
    :param epochs: number of epochs to train the model
    :param epoch_offset: offset to obtain actual epochs in case a checkpoint is restored
    """


def train(
    diffusionModel: cdd_model.DiffusionModel,
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    epochs: int,
    ckpt_manager: tf.train.CheckpointManager,
    globalBatchsize: int,
    epoch_offset: int = 0,
    testFrequency: int = 5,
    checkpointFrequency: int = 20,
    run_output_dir: str = ".",
    callbacks: List[tf.keras.callbacks.Callback] = [],
    distributedTraining: bool = False,
) -> None:
    """Training loop.

    Args:
        diffusionModel (cdd_model.DiffusionModel): Abstraction of the diffusion model.
        train_ds (tf.data.Dataset): Training dataset.
        test_ds (tf.data.Dataset): Testing dataset.
        epochs (int): Number of epochs to train.
        ckpt_manager (tf.train.CheckpointManager): Checkpoint manager instance.
        globalBatchsize (int): Batch size considering all workers running in parallel in a data parallel setup.
        epoch_offset (int, optional): Epoch to start with. Defaults to 0.
        testFrequency (int, optional): Number of epochs to pass between test steps. Defaults to 5.
        checkpointFrequency (int, optional): Number of epochs to pass between checkpoint creation. Defaults to 20.
        run_output_dir (str, optional): Directory where results are saved. Defaults to ".".
        callbacks (List[tf.keras.callbacks.Callback], optional): Callbacks to be triggered. Defaults to [].
        distributedTraining (bool, optional): If true, distributed training is executed. Defaults to False.
    """
    if distributedTraining:
        train_ds = diffusionModel.strategy.experimental_distribute_dataset(train_ds)
        test_ds = diffusionModel.strategy.experimental_distribute_dataset(test_ds)
        train_step_func = diffusionModel.distributed_train_step
        test_step_func = diffusionModel.distributed_test_step
    else:
        train_step_func = diffusionModel.train_step
        test_step_func = diffusionModel.test_step

    for callback in callbacks:
        callback.on_train_begin()

    train_num_batches = "???"
    test_num_batches = "???"
    # start training timer
    training_start_time = time.time()
    logging.info("Start trainings loop...")

    epochs += epoch_offset

    for epoch in range(epoch_offset + 1, epochs + 1):  # from 1 to epochs+1, considering a potential offset.
        # start epoch timer
        epoch_time = 0.0
        epoch_start_time = time.time()
        train_losses = []

        for callback in callbacks:
            callback.on_train_step_begin(epoch)

        for i, batched_x0 in enumerate(iter(train_ds)):
            train_loss = train_step_func(batched_x0, globalBatchsize)
            train_losses.append(train_loss)
            epoch_time = round(time.time() - epoch_start_time, 3)
            print(
                bcolors.OKBLUE
                + f"\rEpoch[{epoch}|{epochs}](train) - Batch[{i}|{train_num_batches}] - Epoch time: {int(epoch_time/60):3} min {int(epoch_time%60):2} sec - Batch loss: {train_loss:.6f}"
                + bcolors.ENDC,
                end="",
            )

        avg_train_loss = np.mean(train_losses).item()

        if train_num_batches == "???":
            train_num_batches = str(len(train_losses))

        total_time = round(time.time() - training_start_time, 3)
        print(
            bcolors.OKBLUE
            + f"\rEpoch[{epoch}|{epochs}](train) - Epoch time: {int(epoch_time/60):3} min {int(epoch_time%60):2} sec - Total time: {int(total_time/3600):2} h {int((total_time%3600)/60):3} min {int(total_time%60):2} sec - Avg. Loss = {avg_train_loss:.6f}"
            + bcolors.ENDC
        )

        with open(os.path.join(run_output_dir, "train_loss.yml"), "a") as outfile:
            yaml.dump({epoch: avg_train_loss}, outfile)

        for callback in callbacks:
            callback.on_train_step_end(epoch)

        if testFrequency and (epoch % testFrequency == 0) or epoch == epochs:  # test
            test_start_time = time.time()
            test_losses = []
            for i, batched_x0 in enumerate(iter(test_ds)):
                test_loss = test_step_func(batched_x0, globalBatchsize)
                test_losses.append(test_loss)
                epoch_time = round(time.time() - test_start_time, 3)
                print(
                    bcolors.OKGREEN
                    + f"\rEpoch[{epoch}|{epochs}](test)  - Batch[{i}|{test_num_batches}] - Epoch time: {int(epoch_time/60):3} min {int(epoch_time%60):2} sec - Batch loss: {test_loss:.6f}"
                    + bcolors.ENDC,
                    end="",
                )

            avg_test_loss = np.mean(test_losses).item()
            # calculate training time
            if test_num_batches == "???":
                test_num_batches = str(len(test_losses))
            total_time = round(time.time() - training_start_time, 3)
            print(
                bcolors.OKGREEN
                + f"\rEpoch[{epoch}|{epochs}](test)  - Epoch time: {int(epoch_time/60):3} min {int(epoch_time%60):2} sec - Total time: {int(total_time/3600):2} h {int((total_time%3600)/60):3} min {int(total_time%60):2} sec - Avg. Loss = {avg_test_loss:.6f}"
                + bcolors.ENDC
            )

            with open(os.path.join(run_output_dir, "test_loss.yml"), "a") as outfile:
                yaml.dump({epoch: avg_test_loss}, outfile)

        if epoch % checkpointFrequency == 0 or epoch == epochs:
            ckpt_manager.save(checkpoint_number=epoch)

        for callback in callbacks:
            callback.on_epoch_end(epoch)

    for callback in callbacks:
        callback.on_train_end()


def eval(
    diffusionModel: cdd_model.DiffusionModel,
    dataset: tf.data.Dataset,
    globalBatchsize: int,
    output_dir: str,
    distributedEval: bool = False,
) -> None:
    """Evaluation loop.

    Args:
        diffusionModel (cdd_model.DiffusionModel): Abstraction of the diffusion model.
        dataset (tf.data.Dataset): Dataset used for evaluation.
        globalBatchsize (int): Batch size considering all workers running in parallel in a data parallel setup.
        output_dir (str): Directory where results are saved.
        distributedEval (bool, optional): If true, distributed evaluation is executed. Defaults to False.
    """
    if distributedEval:
        dataset = diffusionModel.strategy.experimental_distribute_dataset(dataset)
        eval_step_func = diffusionModel.distributed_eval_step
    else:
        eval_step_func = diffusionModel.eval_step

    VLB_list = []
    mae_list = []
    mse_list = []
    iou_list = []
    dice_list = []
    x_translationlist = []
    y_translation_list = []
    mae_shifted_list = []
    mse_shifted_list = []
    iou_shifted_list = []
    dice_shifted_list = []
    for i, batched_x0 in enumerate(iter(dataset)):
        logging.info(f"Batch {i+1}")
        metrics = eval_step_func(batched_x0, globalBatchsize)
        VLB_list.append(metrics[0])
        mae_list.append(metrics[1])
        mse_list.append(metrics[2])
        iou_list.append(metrics[3])
        dice_list.append(metrics[4])
        x_translationlist.append(metrics[5])
        y_translation_list.append(metrics[6])
        mae_shifted_list.append(metrics[7])
        mse_shifted_list.append(metrics[8])
        iou_shifted_list.append(metrics[9])
        dice_shifted_list.append(metrics[10])

    # Here reduce mean over all batches (NOT the collection of the distributed batches), .item() is used in order to dump float values to output metrics.yml
    VLB = np.mean(VLB_list).item()
    mae = np.mean(mae_list).item()
    mse = np.mean(mse_list).item()
    iou = np.mean(iou_list).item()
    dice = np.mean(dice_list).item()
    x_translation = np.mean(x_translationlist).item()
    y_translation = np.mean(y_translation_list).item()
    mae_shifted = np.mean(mae_shifted_list).item()
    mse_shifted = np.mean(mse_shifted_list).item()
    iou_shifted = np.mean(iou_shifted_list).item()
    dice_shifted = np.mean(dice_shifted_list).item()

    logging.info("### FINAL RESULT ###")
    logging.info(
        f"VLB: {VLB:.4f} MAE: {mae:.4f} MSE: {mse:.4f} IoU: {iou:.4f} Dice: {dice:.4f} x_translation: {x_translation:.4f} y_translation: {y_translation:.4f} MAE shifted: {mae_shifted:.4f} MSE shifted: {mse_shifted:.4f} IoU shifted: {iou_shifted:.4f} Dice shifted: {dice_shifted:.4f}"
    )

    with open(os.path.join(output_dir, "metrics.yml"), "a") as outfile:
        yaml.dump(
            {
                "VLB": VLB,
                "MAE": mae,
                "MSE": mse,
                "IoU": iou,
                "Dice": dice,
                "x_translation": x_translation,
                "y_translation": y_translation,
                "MAE_shifted": mae_shifted,
                "MSE_shifted": mse_shifted,
                "iou_shifted": iou_shifted,
                "dice_shifted": dice_shifted,
            },
            outfile,
        )


def sample(
    diffusionModel: cdd_model.DiffusionModel,
    dataset: tf.data.Dataset,
    output_dir: str,
    sampler: str,
    sampling_steps: int,
    diffusionFormat: str,
    conditionFormat: str,
    plot_output: bool = False,
    threshold: float = -0.9,
) -> None:
    """Sample a diffusion model.

    Args:
        diffusionModel (cdd_model.DiffusionModel): Abstraction of the diffusion model.
        dataset (tf.data.Dataset): Dataset containing condition inputs.
        output_dir (str): Directory where results are saved
        sampler (str): Sampling algorithm: [ ddpm | ddim ]
        sampling_steps (int): Number of time steps used for sampling.
        diffusionFormat (str): Data format of the diffusion input: [ rgb | depth | rgbd ]
        conditionFormat (str): Data format of the condition input: [ rgb | depth | rgbd ]
        plot_output (bool, optional): If true, outputs are plotted and saved. Defaults to False.
        threshold (float, optional): Threshold for removing background pixel. Defaults to -0.9.

    Raises:
        ValueError: if `sampler` is not supported.
    """
    assert sampler in ["ddpm", "ddim"], f"Sampler '{sampler}' not supported."
    if sampler == "ddpm":
        sampler_func = diffusionModel.DDPMSampler
    elif sampler == "ddim":
        sampler_func = diffusionModel.DDIMSampler

    if plot_output:
        file_path = os.path.join(output_dir, "plots")
        if not os.path.exists(file_path):
            os.mkdir(file_path)

    for i, batched_x0 in enumerate(iter(dataset)):
        logging.info(f"Batch {i+1}")
        output = sampler_func(batched_x0, sampling_steps, threshold=threshold)
        # Save Diffuse data
        SaveDiffusedSamples(
            output["x0"], output_dir, diffusionFormat, fileNamePrefix=f"diff_{sampler}_{sampling_steps}_batch{i}"
        )
        # Save corresponding Condition!
        SaveDiffusedSamples(
            batched_x0, output_dir, conditionFormat, fileNamePrefix=f"cond_{sampler}_{sampling_steps}_batch{i}"
        )

        # TODO: add support for superres color
        if plot_output and diffusionFormat == "depth":
            if conditionFormat == "rgb":  # Plot colored point cloud
                for sample, (depth, img) in enumerate(zip(output["x0"], batched_x0)):
                    cdd_plotting.PlotDualPointCloud(
                        depth,
                        depth,
                        img,
                        hide_axis_and_grid=True,
                        drop_shaddow=True,
                        threshold=threshold,
                        place_on_ground=True,
                        fileName=os.path.join(file_path, f"pc_batch{i}_sample{sample}") if plot_output else None,
                    )
                    cdd_plotting.PlotDepthMap(
                        depth,
                        threshold=threshold,
                        fileName=os.path.join(file_path, f"depthmap_batch{i}_sample{sample}") if plot_output else None,
                    )
            else:  # plot only point cloud.
                for sample, (depth, img) in enumerate(zip(output["x0"], batched_x0)):
                    cdd_plotting.PlotPointCloud(
                        depth,
                        hide_axis_and_grid=True,
                        drop_shaddow=True,
                        threshold=threshold,
                        place_on_ground=True,
                        fileName=os.path.join(file_path, f"pc_batch{i}_sample{sample}") if plot_output else None,
                    )
                    cdd_plotting.PlotDepthMap(
                        depth,
                        threshold=threshold,
                        fileName=os.path.join(file_path, f"depthmap_batch{i}_sample{sample}") if plot_output else None,
                    )


def GetCallbacks(
    model: cdd_model.DiffusionModel,
    CONFIG: Dict[str, Any],
    test_ds: tf.data.Dataset,
) -> List[tf.keras.callbacks.Callback]:
    """Get Callbacks based on `CONFIG`.

    Args:
        model (cdd_model.DiffusionModel): Abstraction of a diffusion model.
        CONFIG (Dict[str, Any]): Configuration dictionary.
        test_ds (tf.data.Dataset): Dataset used for testing.

    Returns:
        List[tf.keras.callbacks.Callback]: List containing callback instances.
    """
    callbacks = []

    if True:
        modelInfoCallback = cdd_callbacks.ModelInfoCallback(model=model, logDir=CONFIG["OUTDIR"])
        callbacks.append(modelInfoCallback)

    if CONFIG["SAMPLE_FREQUENCY"] > 0:
        sampleModelCallback = cdd_callbacks.SampleModelCallback(
            model=model,
            sampling_steps=CONFIG["SAMPLING_STEPS"],
            sampleFrequency=CONFIG["SAMPLE_FREQUENCY"],
            epochs=CONFIG["EPOCHS"],
            scope=CONFIG["SCOPE"],
            condition_format=CONFIG["CONDITION_FORMAT"],
            diffusion_format=CONFIG["DIFFUSION_FORMAT"],
            dataset=test_ds,
            run_output_dir=CONFIG["OUTDIR"],
        )
        callbacks.append(sampleModelCallback)

    if CONFIG["WARM_UP_EPOCHS"] > 0 or CONFIG["LR_DECAY"]:
        learningRate_Callback = cdd_callbacks.LearningRateCallback(
            learning_rate=CONFIG["LEARNING_RATE"],
            model=model,
            logDir=CONFIG["OUTDIR"],
            warmup_epochs=CONFIG["WARM_UP_EPOCHS"],
            epochs=CONFIG["EPOCHS"],
            lr_decay=CONFIG["LR_DECAY"],
            weight_decay=CONFIG["WEIGHT_DECAY"] if CONFIG["OPTIMIZER"] in ["adamW", "sgdW"] else None,
        )
        callbacks.append(learningRate_Callback)

    return callbacks


def calc_depth_metrics(
    depth_gt: tf.Tensor,
    depth_pred: tf.Tensor,
    threshold: float = -0.9,
) -> Dict[str, float]:
    """Calculate metrics for generated depth maps.

    Metrics are:
    * IoU: intersection over union.
    * dice: Dice score.
    * mae: Mean average error.
    * mse: Mean square error.

    Args:
        depth_gt (tf.Tensor): Ground truth depth map.
        depth_pred (tf.Tensor): Predicted depth map.
        threshold (float, optional): Threshold for removing background pixel. Defaults to -0.9.

    Returns:
        Dict[str, float]: Dictionary containing the metrics. Keys are: `iou`, `dice`, `mae` and `mse`
    """
    metrics = {}
    mask_gt = tf.squeeze(tf.where(depth_gt > threshold, 1, 0), axis=-1)
    mask_pred = tf.squeeze(tf.where(depth_pred > threshold, 1, 0), axis=-1)
    intersection = tf.where((mask_gt * mask_pred) > 0, 1, 0)
    union = tf.where((mask_gt + mask_pred) > 0, 1, 0)

    # count values for each batch
    num_intersection = tf.math.reduce_sum(
        intersection, axis=[1, 2]
    )  # values are either 1 or 0 so summing all values returns number of values that are 1
    num_union = tf.math.reduce_sum(
        union, axis=[1, 2]
    )  # values are either 1 or 0 so summing all values returns number of values that are 1
    num_mask_gt = tf.math.reduce_sum(
        mask_gt, axis=[1, 2]
    )  # values are either 1 or 0 so summing all values returns number of values that are 1
    num_mask_pred = tf.math.reduce_sum(
        mask_pred, axis=[1, 2]
    )  # values are either 1 or 0 so summing all values returns number of values that are 1

    # calc metrics for each batch
    metrics["iou"] = num_intersection / num_union
    metrics["dice"] = 2 * num_intersection / (num_mask_gt + num_mask_pred)
    metrics["mae"] = tf.math.reduce_mean(tf.math.abs(depth_gt - depth_pred), axis=[1, 2, 3])
    metrics["mse"] = tf.math.reduce_mean((depth_gt - depth_pred) ** 2, axis=[1, 2, 3])

    return metrics


def get_shift(
    depth_gt: Union[tf.Tensor, np.ndarray],
    depth_pred: Union[tf.Tensor, np.ndarray],
    threshold: float = -0.9,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Estimate the shift in x,y between two depth maps using 2D correlation.

    Args:
        depth_gt (Union[tf.Tensor, np.ndarray]): Ground truth depth map.
        depth_pred (Union[tf.Tensor, np.ndarray]): Predicted depth map.
        threshold (float, optional): Threshold for removing background pixel. Defaults to -0.9.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Estimated translation in `y` and `x` and the `shifted_depth_map`.
    """
    depth_pred = tf.convert_to_tensor(depth_pred)  # enforce to be tf tensor. input might be numpy array
    batch, height, width, channel = tf.shape(depth_pred)
    mask_gt = tf.squeeze(tf.where(depth_gt > threshold, 1, 0), axis=-1)
    mask_pred = tf.squeeze(tf.where(depth_pred > threshold, 1, 0), axis=-1)

    # get correlation. no batched implementation available, hence the loop.
    # TODO: ugly batch implementation... for now it just must work!
    shifted_depth_pred = depth_pred.numpy()
    x = []
    y = []
    for b in range(batch):
        correlation = signal.correlate2d(mask_gt[b, ...], mask_pred[b, ...], mode="same")
        _y, _x = np.unravel_index(np.argmax(correlation), correlation.shape)
        y_shift = _y - height // 2
        x_shift = _x - width // 2
        y.append(y_shift)
        x.append(x_shift)
        shifted_depth_pred[b, ...] = np.roll(shifted_depth_pred[b, ...], y_shift, axis=0)
        shifted_depth_pred[b, ...] = np.roll(shifted_depth_pred[b, ...], x_shift, axis=1)

    y = tf.cast(y, dtype=tf.float32)
    x = tf.cast(x, dtype=tf.float32)
    return y, x, shifted_depth_pred


def get_optimizer(optimizer: str, learning_rate: float, weight_decay: float = 0.0) -> tf.keras.optimizers.Optimizer:
    """Get optimizer instance.

    Args:
        optimizer (str): String describing the optimizer: [ adam | adamW | sgd | sgdW | yogi ]
        learning_rate (float): Learning rate for the optimization step.
        weight_decay (float, optional): _description_. Defaults to 0.0.

    Raises:
        ValueError: if `optimizer` is not supported.

    Returns:
        tf.keras.optimizers.Optimizer: An optimizer instance.
    """
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "adamW":
        optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay, learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == "sgdW":
        optimizer = tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=learning_rate, momentum=0.9)
    elif optimizer == "yogi":
        optimizer = tfa.optimizers.Yogi(learning_rate=learning_rate)
    else:
        raise ValueError(f"Undefined optimizer provided: {optimizer}")
    return optimizer
