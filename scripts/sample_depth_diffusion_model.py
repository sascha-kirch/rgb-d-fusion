import argparse

# fmt: off
parser = argparse.ArgumentParser(description="Sample from a trained depth diffusion model providing RGB conditions as input")
parser.add_argument("-s","--sampling_steps", type=int, default=None, help="Number of steps used for sampling. If None, use number of steps used during training.")
parser.add_argument("-b", "--batch_size", type=int, default=16, help="Number of samples that are put throught the model simultaniously. The bigger the more memory is required by the GPU.")
parser.add_argument("-n", "--number_samples", type=int, default=None, help="Number of samples to be loaded from the dataset. If None or if -n is larger than the available sample, the entire set is loaded.")
parser.add_argument("-sa", "--sampler", default="ddpm", choices=["ddpm","ddim"], help="The dataset used for evaluation")
parser.add_argument("-df", "--data_format", default="depth", choices=["rgb","rgbd","depth"], help="The format of the diffused samples. E.g. if the output is a depthmap, use depth.")
parser.add_argument("-a", "--xla_accelerate",action="store_true", help="Whether or not to accelerate using XLA for algebraic computations")
parser.add_argument("-m", "--mixed_precission",action="store_true", help="Whether or not to use mixed precission")
parser.add_argument("-p", "--plot_samples",action="store_true", help="Whether or not to plot samples")
parser.add_argument("-bd","--base_dir", default="/tf", help="Base directory for saving output directories and loading datasets from")
parser.add_argument("images_base_dir", help="Path to the base directory of the images that should be used as condition for sampling. The base directory should contain a RENDER directory for the RGB images and DEPTH_RENDER_EXR for depth data.")
parser.add_argument("checkpoint", help="Path to the checkpoint that should be restored.")
parser.add_argument("config_file", help="Path to the configfile that has been generated during training")
args = parser.parse_args()
# fmt: on

import logging

logging.basicConfig(level="INFO", format="[%(levelname)s | %(asctime)s] - %(message)s", datefmt="%I:%M:%S %p")
logging.info("---------------------------------------------------------")
logging.info(f"Running {__file__} ")
logging.info("---------------------------------------------------------")
logging.debug(args)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only print errors, not warnings or infos
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Enable EXR

import sys

# Get the parent directory to be able to import the files located in imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import subprocess
import time

import DeepSaki
import tensorflow as tf
import yaml

import imports.dataset as cdd_dataset
import imports.helper as cdd_helper
import imports.model as cdd_model

os.chdir(args.base_dir)
assert os.path.isfile(args.checkpoint + ".index"), f"No such file: '{args.checkpoint}'"
assert os.path.isfile(args.config_file), f"No such file: '{args.config_file}'"

# reload training config
logging.info(f"Loading config file: {args.config_file}")
with open(args.config_file, "r") as file:
    CONFIG = yaml.load(file, Loader=yaml.FullLoader)

# If argument is none, use the one from config file, otherwise use argument
if not args.sampling_steps:
    logging.info(f"No sampling steps provided, using sampling steps from config file: {CONFIG['SAMPLING_STEPS']}")
    args.sampling_steps = CONFIG["SAMPLING_STEPS"]

# set random seed
RANDOM_SEED = 1911
tf.random.set_seed(RANDOM_SEED)

strategy, _, hw_accelerator_handle = DeepSaki.utils.DetectHw()
CONFIG["GLOBAL_BATCH_SIZE"] = args.batch_size * strategy.num_replicas_in_sync
CONFIG["DISTRIBUTED_EVAL"] = isinstance(strategy, tf.distribute.MirroredStrategy)

if args.mixed_precission:
    DeepSaki.utils.EnableMixedPrecision()

if args.xla_accelerate:
    DeepSaki.utils.EnableXlaAcceleration()

# extract directoryname from configfile
CONFIG["OUTDIR"] = os.path.join(
    os.path.dirname(args.config_file),
    "diffusion_output",
    "{timestamp}_{sampler}_{samplingsteps}".format(
        timestamp=int(time.time()), sampler=args.sampler, samplingsteps=args.sampling_steps
    ),
)
subprocess.run(["mkdir", "-p", CONFIG["OUTDIR"]], check=True)

# save config
logging.info(f"Saving data to: {CONFIG['OUTDIR']}")
with open(os.path.join(CONFIG["OUTDIR"], "CONFIG.yml"), "w") as outfile:
    yaml.dump(args, outfile)

#############################################################
###### Load Images
#############################################################
policy = tf.keras.mixed_precision.global_policy()

dataset = cdd_dataset.GetDatasetDepthDiffusionStreamedForSampling(
    datasetDirectory=args.images_base_dir,
    batchSize=CONFIG["GLOBAL_BATCH_SIZE"],
    img_height_width=CONFIG["IMG_HEIGHT_WIDTH"],
    NumberOfSamplesToRead=args.number_samples,
    dtype=policy.variable_dtype,
    cropWidthHalf=CONFIG["CROP_WIDTH_HALF"],
)


#############################################################
###### Reload Model Checkpoint
#############################################################
optimizer = cdd_helper.get_optimizer(CONFIG["OPTIMIZER"], CONFIG["LEARNING_RATE"], CONFIG["WEIGHT_DECAY"])

# Support for older models
if "DIFFUSION_FORMAT" in CONFIG:
    diffusion_format = CONFIG["DIFFUSION_FORMAT"]
else:
    logging.warning(
        "'DIFFUSION_FORMAT' not found in CONFIG. This might be because you are running evaluation on an older trained model. Setting diffusion format to 'depth'"
    )
    diffusion_format = "depth"

if "CONDITION_FORMAT" in CONFIG:
    condition_format = CONFIG["CONDITION_FORMAT"]
else:
    logging.warning(
        "'CONDITION_FORMAT' not found in CONFIG. This might be because you are running evaluation on an older trained model. Setting condition format to 'rgb'"
    )
    condition_format = "rgb"

with strategy.scope():
    if CONFIG["MODEL"] == "unet":
        model = cdd_model.ConditionalUnet(
            baseDim=CONFIG["BASE_DIM"],
            dim_mults=CONFIG["DIM_MULTIPLIER"],
            numBlocks=CONFIG["NUM_BLOCKS"],
            numResBlocks=CONFIG["NUM_RES_BLOCKS"],
            attentionBlocks=CONFIG["ATTENTION_BLOCKS"],
            dropResBlockRate=CONFIG["DROP_RESBLOCK_RATE"],
            diffusionChannels=cdd_helper.get_channels_from_format(diffusion_format),
            learned_variance=CONFIG["VARIANCE_TYPE"] in ["learned", "learned_range"],
            downsampling=CONFIG["DOWN_SAMPLING"],
        )
    elif CONFIG["MODEL"] == "unet3+":
        model = cdd_model.Unet3plus(
            baseDim=CONFIG["BASE_DIM"],
            dim_mults=CONFIG["DIM_MULTIPLIER"],
            numBlocks=CONFIG["NUM_BLOCKS"],
            numResBlocks=CONFIG["NUM_RES_BLOCKS"],
            attentionBlocks=CONFIG["ATTENTION_BLOCKS"],
            dropResBlockRate=CONFIG["DROP_RESBLOCK_RATE"],
            concat_filters=CONFIG["BASE_DIM"],
            diffusionChannels=cdd_helper.get_channels_from_format(diffusion_format),
            learned_variance=CONFIG["VARIANCE_TYPE"] in ["learned", "learned_range"],
            downsampling=CONFIG["DOWN_SAMPLING"],
        )
    else:
        raise Exception(f'Undefined model provided: {CONFIG["MODEL"]}')

    diffusionModel = cdd_model.DiffusionModel(
        model=model,
        varianceType=CONFIG["VARIANCE_TYPE"],
        diffusionSteps=CONFIG["TIMESTEPS"],
        betaScheduleType=CONFIG["BETA_SCHEDULE"],
        lossWeighting=CONFIG["LOSS_WEIGHTING_TYPE"],
        lambdaVLB=CONFIG["LAMBDA_L_VLB"],
        mixedPrecission=CONFIG["USE_MIXED_PRECISION"],
        diffusionInputShapeChannels=cdd_helper.get_channels_from_format(diffusion_format),
        diffusionInputShapeHeightWidth=CONFIG["IMG_HEIGHT_WIDTH"],
    )
    diffusionModel.compile(optimizer)
    diffusionModel.model.build(
        (*list(CONFIG["IMG_HEIGHT_WIDTH"]), cdd_helper.get_channels_from_format(condition_format)),
        (*list(CONFIG["IMG_HEIGHT_WIDTH"]), cdd_helper.get_channels_from_format(diffusion_format)),
    )

    diffusionModel.model.summary(line_length=100, expand_nested=True)

    # Load checkpoint
    ckpt = tf.train.Checkpoint(model=diffusionModel.model, optimizer=diffusionModel.optimizer)
    ckpt.restore(args.checkpoint)
    print(f"Restored model from {args.checkpoint}")


#############################################################
###### Sampling
#############################################################

logging.info(f"Start sampling model using {args.sampler} sampler...")
cdd_helper.sample(
    diffusionModel,
    dataset=dataset,
    output_dir=CONFIG["OUTDIR"],
    sampler=args.sampler,
    sampling_steps=args.sampling_steps,
    diffusionFormat=diffusion_format,
    conditionFormat=condition_format,
    plot_output=args.plot_samples,
)
