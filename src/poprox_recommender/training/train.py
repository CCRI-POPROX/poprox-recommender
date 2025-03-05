import argparse
import importlib

import torch
from lenskit.logging import LoggingConfig, get_logger
from safetensors.torch import load_file
from transformers import Trainer, TrainingArguments

from poprox_recommender.config import default_device
from poprox_recommender.evaluation import evaluate
from poprox_recommender.paths import project_root
from poprox_recommender.training.dataset import BaseDataset, ValDataset

logger = get_logger("poprox_recommender.training.train")
root = project_root()


def train(device, load_checkpoint):
    """
    1. Initialize model
    """
    if not device.startswith("cuda"):
        logger.warning("training on %s, not CUDA", device)

    logger.info("Initialize Model")
    Model = getattr(importlib.import_module("poprox_recommender.model.nrms"), "NRMS")
    model = Model(args).to(device)

    if load_checkpoint:
        checkpoint = load_file(args.checkpoint_path)
        model.load_state_dict(checkpoint)

    """
    def print_model_size(model):
        print("Model: ", model.__class__.__name__)
        total_params = 0
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
            total_params += param.numel()
        print(f"Total number of parameters in the model: {total_params}")
    """

    """
    2. Load Data & Create Dataset
    """
    logger.info("Initialize Dataset")
    # train_dataset = root / "data/MINDlarge_post_train/behaviors_parsed.tsv"
    train_dataset = BaseDataset(
        args,
        root / "data/MINDlarge_post_train/behaviors_parsed.tsv",
        root / "data/MINDlarge_post_train/news_parsed.tsv",
    )
    logger.info(f"The size of train_dataset is {len(train_dataset)}.")
    # eval_dataset = root / "data/MINDlarge_dev/behaviors.tsv"
    eval_dataset = ValDataset(
        args, root / "data/MINDlarge_dev/behaviors.tsv", root / "data/MINDlarge_post_dev/news_parsed.tsv"
    )
    logger.info(f"The size of eval_dataset is {len(eval_dataset)}.")

    """
    3. Train
    """
    logger.info("Training Start")
    training_args = TrainingArguments(
        output_dir="models/nrms-mind",
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
        logging_dir="models/nrms-mind",
        logging_steps=1,
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
    4. Evaluate model by validation dataset
    """
    logger.info("Evaluation")
    auc, mrr, ndcg5, ndcg10 = evaluate(args, model, device)
    logger.info(f"evaluation auc={auc}, mrr={mrr}, ndcg5={ndcg5}, ndcg10={ndcg10}")


if __name__ == "__main__":
    import os  # noqa: F401

    device = default_device()

    """
    The followings need modification based on need
    More detailed hyper-parameters for model training need to be modified in TrainingArguments
    """
    parser = argparse.ArgumentParser(description="processing some parameters")

    parser.add_argument("-v", "--verbose", help="enable verbose logging")
    parser.add_argument("--num_clicked_news_a_user", type=float, default=50)  # length of clicked history
    parser.add_argument("--dataset_attributes", type=str, default="title")
    parser.add_argument("--dropout_probability", type=float, default=0.2)
    parser.add_argument("--num_attention_heads", type=float, default=16)  # for newsencoder
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
    lc = LoggingConfig()
    if args.verbose:
        lc.set_verbose(True)
    lc.apply()

    torch.cuda.empty_cache()

    train(device, args.load_checkpoint)
