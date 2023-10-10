import logging
import os
import sys


# Get the parent directory to be able to import the files located in imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import tensorflow as tf

import imports.helper as cdd_helper
import imports.plotting as cdd_plotting
import yaml


class ModelInfoCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, logDir):
        self.model = model
        self.logDir = logDir

    def on_train_begin(self, logs=None):
        fileName = os.path.join(self.logDir, "illustrations", "model.png")
        self.model.model.PlotGraph(fileName)

        # Print Summary to file
        with open(os.path.join(self.logDir, "model_summary.txt"), "w") as fh:
            self.model.model.summary(print_fn=lambda x: fh.write(x + "\n"), line_length=100, expand_nested=True)

    # REQUIRED in order for tracing and graping to work only for the test data, since parent does not implements this!
    def on_train_step_begin(self, epoch, logs=None):
        pass

    def on_train_step_end(self, epoch, logs=None):
        pass


class SampleModelCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model,
        sampling_steps,
        sampleFrequency,
        epochs,
        scope,
        condition_format,
        diffusion_format,
        dataset,
        run_output_dir,
    ):
        self.model = model
        self.sampling_steps = sampling_steps
        self.sampleFrequency = sampleFrequency
        self.epochs = epochs
        self.dataset = dataset
        self.scope = scope
        self.condition_format = condition_format
        self.diffusion_format = diffusion_format
        self.run_output_dir = run_output_dir

    def on_train_begin(self, logs=None):
        pass

    # REQUIRED in order for tracing and graping to work only for the test data, since parent does not implements this!
    def on_train_step_begin(self, epoch, logs=None):
        pass

    def on_train_step_end(self, epoch, logs=None):
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


class LearningRateCallback(tf.keras.callbacks.Callback):
    """
    A callback to change the learning rate of the optimizers.
    """

    def __init__(self, learning_rate, model, logDir, warmup_epochs, epochs, weight_decay=None, lr_decay=None):
        self.learning_rate = learning_rate
        self.model = model
        self.logDir = logDir
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.decayFunc = None
        if lr_decay == None:
            self.decayFunc = self.NoDecay
        elif lr_decay == "linear":
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
            raise Exception(f"provided lr_decay: '{lr_decay}' is not defined")

    def LinearWarmUp(self, target_lr, warmup_epochs, currentEpoch):
        return (currentEpoch + 1) * (target_lr / (warmup_epochs + 1))

    def LinearDecay(self, currentEpoch, warmup_epochs, total_epochs, target_lr):
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        else:
            return -(currentEpoch - warmup_epochs) * (target_lr / (total_epochs - (warmup_epochs))) + target_lr

    def ExponentialDecay(self, currentEpoch, warmup_epochs, total_epochs, target_lr):
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        else:
            k = 2 * np.pi / (total_epochs - warmup_epochs)
            return target_lr * np.exp(-k * (currentEpoch + 1 - (warmup_epochs + 1)))

    def StepDecay(self, currentEpoch, warmup_epochs, total_epochs, target_lr):
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        else:
            quarter = (total_epochs - warmup_epochs) // 4
            if currentEpoch < (warmup_epochs + quarter):
                return target_lr
            elif currentEpoch < (warmup_epochs + 2 * quarter):
                return 0.5 * target_lr
            elif currentEpoch < (warmup_epochs + 3 * quarter):
                return 0.25 * target_lr
            else:
                return 0.125 * target_lr

    def CosineRestartDecay(self, currentEpoch, warmup_epochs, total_epochs, target_lr):
        half = ((total_epochs - warmup_epochs) // 2) + 1
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        else:
            if currentEpoch < (warmup_epochs + half):
                return self.CosineDecay(currentEpoch - warmup_epochs, 0, half, target_lr)
            else:
                return self.CosineDecay(currentEpoch - warmup_epochs - half, 0, half, 0.5 * target_lr)

    def CosineDecay(self, currentEpoch, warmup_epochs, total_epochs, target_lr):
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        else:
            return (target_lr / 2) * np.cos(
                (np.pi / (total_epochs - warmup_epochs)) * (currentEpoch - warmup_epochs)
            ) + (target_lr / 2)

    def NoDecay(self, currentEpoch, warmup_epochs, total_epochs, target_lr):
        if currentEpoch < warmup_epochs:
            return self.LinearWarmUp(target_lr, warmup_epochs, currentEpoch)
        else:
            return target_lr

    def on_train_step_begin(self, epoch, logs=None):
        self.model.optimizer.learning_rate = self.decayFunc(epoch, self.warmup_epochs, self.epochs, self.learning_rate)
        if self.weight_decay:
            self.model.optimizer.weight_decay = self.decayFunc(
                epoch, self.warmup_epochs, self.epochs, self.weight_decay
            )

    def on_train_step_end(self, epoch, logs=None):
        pass
