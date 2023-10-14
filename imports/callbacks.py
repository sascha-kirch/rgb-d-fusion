import logging
import os
import sys
from typing import Optional

# Get the parent directory to be able to import the files located in imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import tensorflow as tf
import yaml

import imports.helper as cdd_helper
import imports.plotting as cdd_plotting
from imports.model import DiffusionModel


class CddCallback(tf.keras.callbacks.Callback):
    """Base class for callbacks. Methods might be overriden, but is not mandatory as in abstract classes.

    Inherits from:
        tf.keras.callbacks.Callback
    """

    def on_train_begin(self, logs: Optional[str] = None) -> None:
        """Event triggered on the beggining of the training.

        Args:
            logs (Optional[str], optional): Extra information that can be provided to the event. Defaults to None.
        """
        pass

    def on_train_step_begin(self, epoch: int, logs: Optional[str] = None) -> None:
        """Event triggered at the beginning of the current training step of a given epoch.

        Args:
            epoch (int): Current epoch in which event is triggered.
            logs (Optional[str], optional): Extra information that can be provided to the event. Defaults to None.
        """
        pass

    def on_train_step_end(self, epoch: int, logs: Optional[str] = None) -> None:
        """Event triggered at the end of the current training step of a given epoch.

        Args:
            epoch (int): Current epoch in which event is triggered.
            logs (Optional[str], optional): Extra information that can be provided to the event. Defaults to None.
        """
        pass


class ModelInfoCallback(CddCallback):
    """Callback to save the model summary in text form and save the model graph as image.

    Event triggered at the beginning of the training before the first training step.

    Inherits from:
        CddCallback
    """

    def __init__(self, model: DiffusionModel, logDir: str) -> None:
        """Callback to save the model summary in text form and save the model graph as image.

        Args:
            model (DiffusionModel): Abstraction of a diffsuion model for which the info shall be saved.
            logDir (str): Directory to which the logs are saved.
        """
        self.model = model
        self.logDir = logDir

    def on_train_begin(self, logs: Optional[str] = None) -> None:
        """Event to save the model summary in text form and save the model graph as image.

        Event triggered at the beginning of the training before the first training step.

        Args:
            logs (Optional[str], optional): Extra information that can be provided to the event. Defaults to None.
        """
        fileName = os.path.join(self.logDir, "illustrations", "model.png")
        self.model.model.PlotGraph(fileName)

        # Print Summary to file
        with open(os.path.join(self.logDir, "model_summary.txt"), "w") as fh:
            self.model.model.summary(print_fn=lambda x: fh.write(x + "\n"), line_length=100, expand_nested=True)


class SampleModelCallback(CddCallback):
    """Callback to sample from the diffusion model, calculate metrics and save plots for evaluation.

    Event triggered after at the end of training step and if the current epoch matches the frequency constraint.

    Inherits from:
        CddCallback
    """

    def __init__(
        self,
        model: DiffusionModel,
        sampling_steps: int,
        sampleFrequency: int,
        epochs: int,
        scope: str,
        condition_format: str,
        diffusion_format: str,
        dataset: tf.data.Dataset,
        run_output_dir: str,
    ) -> None:
        """Callback to sample from the diffusion model, calculate metrics and save plots for evaluation.

        Args:
            model (DiffusionModel): Abstraction of a diffsuion model from which shall be sampled.
            sampling_steps (int): Number of steps in the reverse diffusion process.
            sampleFrequency (int): Number of epochs that should pass before sampling the model.
            epochs (int): Total number of epochs the model is trained.
            scope (str): [ depth_diffusion | super_resolution]
            condition_format (str): Fformat of the condition input of the super resolution model. [rgb | depth | rgbd].
            diffusion_format (str): Format of the diffusion input of the super resolution model. [rgb | depth | rgbd].
            dataset (tf.data.Dataset): Dataset used for sampling.
            run_output_dir (str): Directory where the output metrics and samples are saved to
        """
        self.model = model
        self.sampling_steps = sampling_steps
        self.sampleFrequency = sampleFrequency
        self.epochs = epochs
        self.dataset = dataset
        self.scope = scope
        self.condition_format = condition_format
        self.diffusion_format = diffusion_format
        self.run_output_dir = run_output_dir

    def on_train_step_end(self, epoch: int, logs: Optional[str] = None) -> None:
        """Event to sample from the diffusion model, calculate metrics and save plots for evaluation.

        Event triggered after at the end of training step and if the current epoch matches the frequency constraint.

        Args:
            epoch (int): Current epoch when event is triggered
            logs (Optional[str], optional): Extra information that can be provided to the event. Defaults to None.
        """
        try:
            if self.sampleFrequency and (epoch % self.sampleFrequency == 0) or epoch == self.epochs:
                (
                    x0_condition,
                    x0_diffusion_input,
                ) = (
                    self.dataset.as_numpy_iterator().next()
                )  # get one batch of the dataset with shape (batch, height, width, channel)
                ddpm_output = self.model.DDPMSampler(x0_condition, self.sampling_steps, frames_to_output=50)
                # TODO: incooperate global batchsize in prparation for multi GPU sampling
                # reduce mean consideres sample batch size, not global batchsize, which is ok in this case, because it is calculated over single batch not distributed.
                metrics = cdd_helper.calc_depth_metrics(depth_gt=x0_diffusion_input, depth_pred=ddpm_output["x0"])
                y_shifts, x_shifts, pred_shifted = cdd_helper.get_shift(x0_diffusion_input, ddpm_output["x0"])
                metrics_shifted = cdd_helper.calc_depth_metrics(depth_gt=x0_diffusion_input, depth_pred=pred_shifted)
                mae = tf.math.reduce_sum(metrics["mae"])
                mse = tf.math.reduce_sum(metrics["mse"])
                iou = tf.math.reduce_sum(metrics["iou"])
                dice = tf.math.reduce_sum(metrics["dice"])
                x_translation = tf.math.reduce_sum(tf.math.abs(y_shifts))
                y_translation = tf.math.reduce_sum(tf.math.abs(x_shifts))
                mae_shifted = tf.math.reduce_sum(metrics_shifted["mae"])
                mse_shifted = tf.math.reduce_sum(metrics_shifted["mse"])
                iou_shifted = tf.math.reduce_sum(metrics_shifted["iou"])
                dice_shifted = tf.math.reduce_sum(metrics_shifted["dice"])

                print(
                    f"Epoch[{epoch}|{self.epochs}](sampling) -  MAE: {mae:.4f} - MSE: {mse:.4f} - IoU: {iou:.4f} - Dice: {dice:.4f} - x_translation: {x_translation:.4f} - y_translation: {y_translation:.4f} - MAE shifted: {mae_shifted:.4f} - MSE shifted: {mse_shifted:.4f} - IoU shifted: {iou_shifted:.4f} - Dice shifted: {dice_shifted:.4f}"
                )
                with open(os.path.join(self.run_output_dir, "sample_metrics.yml"), "a") as outfile:
                    yaml.dump(
                        {epoch: [mae, mse, iou, dice, x_translation, y_translation, iou_shifted, dice_shifted]}, outfile
                    )
                if self.scope == "depth_diffusion":
                    cdd_plotting.PlotBatchedSample(
                        x0_condition,
                        x0_diffusion_input,
                        ddpm_output,
                        num_samples=8,
                        run_output_dir=self.run_output_dir,
                        postfix=f"Epoch_{epoch}_DDPM",
                        epoch=epoch,
                    )
                elif self.scope == "super_resolution":
                    cdd_plotting.PlotBatchedSampleSuperRes(
                        x0_condition,
                        x0_diffusion_input,
                        ddpm_output,
                        self.condition_format,
                        self.diffusion_format,
                        num_samples=8,
                        run_output_dir=self.run_output_dir,
                        postfix=f"Epoch_{epoch}_DDPM",
                        epoch=epoch,
                    )
        except Exception as e:
            logging.warning(type(e))
            logging.warning(e.args)
            logging.warning(e)
            logging.warning("Exception during sampling. Continue with training.")


class LearningRateCallback(CddCallback):
    """Callback to schedule the learning rate. Executed on the beginning of a training step.

    Inherits from:
        CddCallback
    """

    def __init__(
        self,
        learning_rate: float,
        model: DiffusionModel,
        logDir: str,
        warmup_epochs: int,
        epochs: int,
        weight_decay: Optional[float] = None,
        lr_decay: Optional[str] = None,
    ) -> None:
        """Callback to schedule the learning rate. Executed on the beginning of a training step.

        Args:
            learning_rate (float): Target learning rate from which the schedule starts after an optional warmup.
            model (DiffusionModel): Abstraction of a diffsuion model that grants access to the optimizer state.
            logDir (str): Directory where the learning rate is logged for each epoch
            warmup_epochs (int): Number of epochs the learning rate is linearly increased
            epochs (int): Number of epochs the model is trained.
            weight_decay (Optional[float], optional): Target wheight decay value that needs to be scheduled along the
                learning rate. Defaults to None.
            lr_decay (Optional[str], optional): Learning rate schedule. Options are
                ['linear'|'cosine'|'cosine_restart'|'step'|'exponential']. Defaults to None.

        Raises:
            ValueError: Unsuported value for lr_decay.
        """
        self.learning_rate = learning_rate
        self.model = model
        self.logDir = logDir
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.decayFunc = self.NoDecay
        if lr_decay == "linear":
            self.decayFunc = self.LinearDecay
        elif lr_decay == "cosine":
            self.decayFunc = self.CosineDecay
        elif lr_decay == "cosine_restart":
            self.decayFunc = self.CosineRestartDecay
        elif lr_decay == "step":
            self.decayFunc = self.StepDecay
        elif lr_decay == "exponential":
            self.decayFunc = self.ExponentialDecay
        else:
            raise ValueError(f"provided lr_decay: '{lr_decay}' is not defined")

    def LinearWarmUp(self, target_lr: float, warmup_epochs: int, currentEpoch: int) -> float:
        """Linear warmup schedule of the learning rate.

        Args:
            target_lr (float): Target learning rate from which the schedule starts.
            warmup_epochs (int): Number of epochs to linearly increase the learning rate from 0 to taget_lr.
            currentEpoch (int): Current epoch of the training process.

        Returns:
            float: Learning rate for the current epoch following the schedule.
        """
        return (currentEpoch + 1) * (target_lr / (warmup_epochs + 1))

    def LinearDecay(self, currentEpoch: int, warmup_epochs: int, total_epochs: int, target_lr: float) -> float:
        """Learning rate schedule with linearly decreasing learning rate after optional linear warmup.

        Args:
            currentEpoch (int): Current epoch of the training process.
            warmup_epochs (int): Number of epochs to linearly increase the learning rate from 0 to taget_lr.
            total_epochs (int): Total number of epochs for the training.
            target_lr (float): Target learning rate from which the schedule starts.

        Returns:
            float: Learning rate for the current epoch following the schedule.
        """
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        return -(currentEpoch - warmup_epochs) * (target_lr / (total_epochs - (warmup_epochs))) + target_lr

    def ExponentialDecay(self, currentEpoch: int, warmup_epochs: int, total_epochs: int, target_lr: float) -> float:
        """Learning rate schedule with exponential decay aftter optional linear warmup.

        Args:
            currentEpoch (int): Current epoch of the training process.
            warmup_epochs (int): Number of epochs to linearly increase the learning rate from 0 to taget_lr.
            total_epochs (int): Total number of epochs for the training.
            target_lr (float): Target learning rate from which the schedule starts.

        Returns:
            float: Learning rate for the current epoch following the schedule.
        """
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        k = 2 * np.pi / (total_epochs - warmup_epochs)
        return target_lr * np.exp(-k * (currentEpoch + 1 - (warmup_epochs + 1)))

    def StepDecay(self, currentEpoch: int, warmup_epochs: int, total_epochs: int, target_lr: float) -> float:
        """Learning rate schedule where amplitude is halved 4 times after optional linear warmup.

        Args:
            currentEpoch (int): Current epoch of the training process.
            warmup_epochs (int): Number of epochs to linearly increase the learning rate from 0 to taget_lr.
            total_epochs (int): Total number of epochs for the training.
            target_lr (float): Target learning rate from which the schedule starts.

        Returns:
            float: Learning rate for the current epoch following the schedule.
        """
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        quarter = (total_epochs - warmup_epochs) // 4
        if currentEpoch < (warmup_epochs + quarter):
            return target_lr
        if currentEpoch < (warmup_epochs + 2 * quarter):
            return 0.5 * target_lr
        if currentEpoch < (warmup_epochs + 3 * quarter):
            return 0.25 * target_lr
        return 0.125 * target_lr

    def CosineRestartDecay(self, currentEpoch: int, warmup_epochs: int, total_epochs: int, target_lr: float) -> float:
        """Cosine shaped learning rate decay with one restart and optional linear warmup.

        Learning rate schedule following a cosine decay that is restarted after it reaches 0 after half the schedule.
        The amplitude after the restart is target_lr/2. Optional linear warmup.

        Args:
            currentEpoch (int): Current epoch of the training process.
            warmup_epochs (int): Number of epochs to linearly increase the learning rate from 0 to taget_lr.
            total_epochs (int): Total number of epochs for the training.
            target_lr (float): Target learning rate from which the schedule starts.

        Returns:
            float: Learning rate for the current epoch following the schedule.
        """
        half = ((total_epochs - warmup_epochs) // 2) + 1
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)

        if currentEpoch < (warmup_epochs + half):
            return self.CosineDecay(currentEpoch - warmup_epochs, 0, half, target_lr)

        return self.CosineDecay(currentEpoch - warmup_epochs - half, 0, half, 0.5 * target_lr)

    def CosineDecay(self, currentEpoch: int, warmup_epochs: int, total_epochs: int, target_lr: float) -> float:
        """Cosine shaped learning rate decay with optional linear warmup.

        Learning rate schedule following a cosine decay. After an optional warmup, the learning reate is decreased from
        target_lr to 0 following a cosine function.

        Args:
            currentEpoch (int): Current epoch of the training process.
            warmup_epochs (int): Number of epochs to linearly increase the learning rate from 0 to taget_lr.
            total_epochs (int): Total number of epochs for the training.
            target_lr (float): Target learning rate from which the schedule starts.

        Returns:
            float: Learning rate for the current epoch following the schedule.
        """
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        return (target_lr / 2) * np.cos((np.pi / (total_epochs - warmup_epochs)) * (currentEpoch - warmup_epochs)) + (
            target_lr / 2
        )

    def NoDecay(self, currentEpoch: int, warmup_epochs: int, total_epochs: int, target_lr: float) -> float:
        """Schedule with constant learning rate and no decay. Optionaly a linear warmup can be scheduled.

        Args:
            currentEpoch (int): Current epoch of the training process.
            warmup_epochs (int): Number of epochs to linearly increase the learning rate from 0 to taget_lr.
            total_epochs (int): Total number of epochs for the training.
            target_lr (float): Target learning rate which is used after the linear warmup.

        Returns:
            float: Learning rate for the current epoch following the schedule.
        """
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        return target_lr

    def on_train_step_begin(self, epoch: int, logs: Optional[str] = None) -> None:
        """Event to update the learning rate following the configured schedule.

        Args:
            epoch (int): Current epoch when event is triggered.
            logs (Optional[str], optional): Extra information that can be provided to the event. Defaults to None.
        """
        self.model.optimizer.learning_rate = self.decayFunc(epoch, self.warmup_epochs, self.epochs, self.learning_rate)
        if self.weight_decay:
            self.model.optimizer.weight_decay = self.decayFunc(
                epoch, self.warmup_epochs, self.epochs, self.weight_decay
            )
