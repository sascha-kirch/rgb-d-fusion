import argparse

# fmt: off
parser = argparse.ArgumentParser(description="Evaluate the Variational Lower Bound (VLB), L1 and L2 distance of a trained depth diffusion model an a dataset")
parser.add_argument("-b", "--batch_size", type=int, default=16, help="Number of samples that are put throught the model simultaniously. The bigger the more memory is required by the GPU.")
parser.add_argument("-n", "--number_samples", type=int, default=None, help="Number of samples to be loaded from the dataset. If None or if -n is larger than the available sample, the entire set is loaded.")
parser.add_argument("-d", "--dataset", default="v_human_rendered", choices=["v_human_rendered","24k_358seg_equalised","19k_316seg_equalised"], help="The dataset used for evaluation")
parser.add_argument("-a", "--xla_accelerate",action="store_true", help="Whether or not to accelerate using XLA for algebraic computations")
parser.add_argument("-m", "--mixed_precission",action="store_true", help="Whether or not to use mixed precission")
parser.add_argument("-bd","--base_dir", default="/tf", help="Base directory for saving output directories and loading datasets from")
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
with open(args.config_file, "r") as file:
    CONFIG = yaml.load(file, Loader=yaml.FullLoader)

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
CONFIG["OUTDIR"] = os.path.dirname(args.config_file)

#############################################################
###### Dataset
#############################################################
policy = tf.keras.mixed_precision.global_policy()

testDirectory = f"manual_datasets/{args.dataset}/test"

test_ds = cdd_dataset.GetDatasetSuperresStreamed(
    datasetDirectory=testDirectory,
    batchSize=CONFIG["GLOBAL_BATCH_SIZE"],
    low_res_height_width=CONFIG["LOW_RES"],
    high_res_height_width=CONFIG["HIGH_RES"],
    condition_format=CONFIG["CONDITION_FORMAT"],
    diffusion_format=CONFIG["DIFFUSION_FORMAT"],
    NumberOfSamplesToRead=args.number_samples,
    dtype=policy.variable_dtype,
    drop_remainder=True,
    cropWidthHalf=CONFIG["CROP_WIDTH_HALF"],
    shuffle=False,
)

#############################################################
###### Reload Model Checkpoint
#############################################################
optimizer = cdd_helper.get_optimizer(CONFIG["OPTIMIZER"], CONFIG["LEARNING_RATE"], CONFIG["WEIGHT_DECAY"])

with strategy.scope():
    if CONFIG["MODEL"] == "unet":
        model = cdd_model.SuperResolutionUnet(
            baseDim=CONFIG["BASE_DIM"],
            dim_mults=CONFIG["DIM_MULTIPLIER"],
            numBlocks=CONFIG["NUM_BLOCKS"],
            numResBlocks=CONFIG["NUM_RES_BLOCKS"],
            attentionBlocks=CONFIG["ATTENTION_BLOCKS"],
            dropResBlockRate=CONFIG["DROP_RESBLOCK_RATE"],
            diffusionChannels=cdd_helper.get_channels_from_format(CONFIG["DIFFUSION_FORMAT"]),
            learned_variance=CONFIG["VARIANCE_TYPE"] in ["learned", "learned_range"],
            upsamplingFactor=CONFIG["RES_MULTIPLIER"],
            downsampling=CONFIG["DOWN_SAMPLING"],
        )
    elif CONFIG["MODEL"] == "unet3+":
        model = cdd_model.SuperResolutionUnet3plus(
            baseDim=CONFIG["BASE_DIM"],
            dim_mults=CONFIG["DIM_MULTIPLIER"],
            numBlocks=CONFIG["NUM_BLOCKS"],
            numResBlocks=CONFIG["NUM_RES_BLOCKS"],
            attentionBlocks=CONFIG["ATTENTION_BLOCKS"],
            dropResBlockRate=CONFIG["DROP_RESBLOCK_RATE"],
            concat_filters=CONFIG["BASE_DIM"],
            diffusionChannels=cdd_helper.get_channels_from_format(CONFIG["DIFFUSION_FORMAT"]),
            learned_variance=CONFIG["VARIANCE_TYPE"] in ["learned", "learned_range"],
            upsamplingFactor=CONFIG["RES_MULTIPLIER"],
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
        diffusionInputShapeChannels=cdd_helper.get_channels_from_format(CONFIG["DIFFUSION_FORMAT"]),
        diffusionInputShapeHeightWidth=CONFIG["HIGH_RES"],
    )
    diffusionModel.compile(optimizer)
    diffusionModel.model.build(
        (*list(CONFIG["LOW_RES"]), cdd_helper.get_channels_from_format(CONFIG["CONDITION_FORMAT"])),
        (*list(CONFIG["HIGH_RES"]), cdd_helper.get_channels_from_format(CONFIG["DIFFUSION_FORMAT"])),
    )

    diffusionModel.model.summary(line_length=100, expand_nested=True)

    # Load checkpoint
    ckpt = tf.train.Checkpoint(model=diffusionModel.model, optimizer=diffusionModel.optimizer)
    ckpt.restore(args.checkpoint)
    print(f"Restored from {args.checkpoint}")


#############################################################
###### Evaluation
#############################################################
cdd_helper.eval(
    diffusionModel=diffusionModel,
    dataset=test_ds,
    globalBatchsize=CONFIG["GLOBAL_BATCH_SIZE"],
    output_dir=CONFIG["OUTDIR"],
    distributedEval=CONFIG["DISTRIBUTED_EVAL"],
)
