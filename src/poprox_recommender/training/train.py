import argparse
from os import fspath

import lenskit
import lenskit.config
import torch
from humanize import metric
from lenskit.logging import LoggingConfig, Task, friendly_duration, get_logger
from safetensors.torch import load_file, save_file
from transformers import Trainer, TrainingArguments

from poprox_recommender.config import default_device
from poprox_recommender.paths import project_root
from poprox_recommender.training.dataset import BaseDataset, ValDataset

logger = get_logger("poprox_recommender.training.train")
root = project_root()

MODEL_DIR = root / "models" / "nrms-mind"


def train(device, args):
    # 1. Initialize model
    if not device.startswith("cuda"):
        logger.warning("training on %s, not CUDA", device)

    logger.info("Initialize Model")
    from poprox_recommender.model.nrms import NRMS as Model

    model = Model(args)

    if args.load_checkpoint:
        checkpoint = load_file(args.checkpoint_path)
        model.load_state_dict(checkpoint)

    model = model.to(device)

    """
    def print_model_size(model):
        print("Model: ", model.__class__.__name__)
        total_params = 0
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
            total_params += param.numel()
        print(f"Total number of parameters in the model: {total_params}")
    """

    # 2. Load and create datasets
    logger.info("Initialize Dataset")
    train_dataset = BaseDataset(
        args,
        root / "data/MINDlarge_post_train/behaviors_parsed.tsv",
        root / "data/MINDlarge_post_train/news_parsed.tsv",
    )
    logger.info(f"The size of train_dataset is {len(train_dataset)}.")
    eval_dataset = ValDataset(
        args, root / "data/MINDlarge_dev/behaviors.tsv", root / "data/MINDlarge_post_dev/news_parsed.tsv"
    )
    logger.info(f"The size of eval_dataset is {len(eval_dataset)}.")

    # 3. Train model
    logger.info("Training Start")
    training_args = TrainingArguments(
        output_dir=fspath(args.output_dir),
        logging_strategy="steps",
        save_total_limit=5,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        optim="adamw_torch",
        save_strategy="epoch",
        eval_strategy="no",
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        remove_unused_columns=False,
        logging_dir=fspath(args.output_dir),
        logging_steps=500,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    """
    torch.autograd.set_detect_anomaly(True)

    # check gradients for NaNs after training
    def check_nan(tensor, name):
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN in {name}: {tensor}")

    for param_name, param_value in model.named_parameters():
        if param_value.grad is not None:
            check_nan(param_value.grad, f"gradient[{param_name}]")
    """

    # 4. save and extract tensors
    save_model(model, args.output_dir)


def save_model(model, output_dir):
    """
    Save a model, both the entire model and extracting the encoders.
    """
    logger.info("saving model", file="model.safetensors")
    save_file(model.state_dict(), output_dir / "model.safetensors")
    logger.info("saving news encoder", file="news_encoder.safetensors")
    save_file(model.news_encoder.state_dict(), output_dir / "news_encoder.safetensors")
    logger.info("saving user encoder", file="user_encoder.safetensors")
    save_file(model.user_encoder.state_dict(), output_dir / "user_encoder.safetensors")


if __name__ == "__main__":
    import os  # noqa: F401

    device = default_device()

    """
    The followings need modification based on need
    More detailed hyper-parameters for model training need to be modified in TrainingArguments
    """
    parser = argparse.ArgumentParser(description="processing some parameters")
    parser.add_argument("--subset", type=int, default=None, help="train on subset")
    parser.add_argument("--output_dir", type=str, default=None, help="where to save the model")
    parser.add_argument("-v", "--verbose", help="enable verbose logging")
    parser.add_argument("--num_clicked_news_a_user", type=float, default=50)  # length of clicked history
    parser.add_argument("--dataset_attributes", type=str, default="title")
    parser.add_argument("--dropout_probability", type=float, default=0.2)
    parser.add_argument("--num_attention_heads", type=int, default=16)  # for newsencoder
    parser.add_argument("--additive_attn_hidden_dim", type=int, default=200)  # for newsencoder
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--evaluate_batch_size", type=int, default=16)
    parser.add_argument("--evaluate_num_worker", type=int, default=4)
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default="models/nrms-mind/model.safetensors")
    """
    The followings need to be consistent with the hyperparameter settings in preprocess.py
    """
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--num_words_title", type=float, default=30)

    args = parser.parse_args()

    # set a seperate output directory for subset
    if args.output_dir is None:
        if args.subset is not None:
            args.output_dir = root / f"models/nrms-mind-subset-{args.subset}"
        else:
            args.output_dir = MODEL_DIR

    os.makedirs(args.output_dir, exist_ok=True)

    lc = LoggingConfig()
    if args.verbose:
        lc.set_verbose(True)
    lc.apply()
    lenskit.configure(project_root())

    torch.cuda.empty_cache()

    with Task("train NRMS") as task:
        task.save_to_file(args.output_dir / "task.json")
        train(device, args)

    logger.info("training completed in %s", friendly_duration(task.duration or -1))
    if task.system_power:
        # divide by 3600 to convert Joules to Wh
        logger.info(
            "training power: %s (%s CPU, %s GPU)",
            metric(task.system_power / 3600, "Wh"),
            metric(task.cpu_power / 3600, "Wh") if task.cpu_power else "unknown",
            metric(task.gpu_power / 3600, "Wh") if task.gpu_power else "unknown",
        )
