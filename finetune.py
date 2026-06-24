#!/usr/bin/env python3

import os
import argparse
import warnings

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from src.model import GenomicDataset, scGrapHiC
from src.utils import create_directory


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    # Run mode
    p.add_argument("--mode", choices=["npz", "raw"], default="npz",
                   help="npz: fine-tune on prebuilt .npz files. raw: not supported "
                        "(run the build step first, then fine-tune on the .npz).")

    # Input data
    p.add_argument("--train_npz", type=str, default=None)
    p.add_argument("--val_npz", type=str, default=None,
                   help="If omitted, 10%% of training data is held out.")
    p.add_argument("--test_npz", type=str, default=None,
                   help="If omitted, validation set is reused.")
    # (raw mode) parsed pseudobulk directories
    p.add_argument("--rnaseq_dir", type=str, default=None)
    p.add_argument("--schic_dir", type=str, default=None)

    # Checkpoint / output
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Pretrained .ckpt whose architecture must match the flags below.")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Resume fine-tuning from this .ckpt.")
    p.add_argument("--output_dir", type=str, default="finetune_output")
    p.add_argument("--experiment", type=str, default="finetune")

    # Architecture flags (must match checkpoint)
    p.add_argument("--rna_seq", action="store_true", default=True)
    p.add_argument("--no_rna_seq", dest="rna_seq", action="store_false")
    p.add_argument("--use_bulk", action="store_true", default=True)
    p.add_argument("--no_use_bulk", dest="use_bulk", action="store_false")
    p.add_argument("--positional_encodings", action="store_true", default=True)
    p.add_argument("--no_positional_encodings", dest="positional_encodings", action="store_false")
    p.add_argument("--ctcf_motif", action="store_true", default=True)
    p.add_argument("--no_ctcf_motif", dest="ctcf_motif", action="store_false")
    p.add_argument("--cpg_motif", action="store_true", default=True)
    p.add_argument("--no_cpg_motif", dest="cpg_motif", action="store_false")
    p.add_argument("--node_features", type=int, default=2)
    p.add_argument("--pos_encodings_dim", type=int, default=16)
    p.add_argument("--encoder_hidden_embedding_size", type=int, default=32)
    p.add_argument("--num_encoder_attn_blocks", type=int, default=4)
    p.add_argument("--num_heads_encoder_attn_blocks", type=int, default=1)
    p.add_argument("--num_graph_conv_blocks", type=int, default=1)
    p.add_argument("--num_graph_encoder_blocks", type=int, default=4)
    p.add_argument("--edge_dims", type=int, default=1)
    p.add_argument("--conv1d_kernel_size", type=int, default=16)
    p.add_argument("--num_decoder_residual_blocks", type=int, default=7)
    p.add_argument("--width", type=int, default=7)
    p.add_argument("--num_channels", type=int, default=1)

    # Dataset preprocessing
    p.add_argument("--resolution", type=int, default=50000)
    p.add_argument("--num_nodes", type=int, default=128)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--bounds", type=int, default=10)
    p.add_argument("--padding", type=bool, default=True)
    p.add_argument("--remove_borders", type=int, default=30_000_000)
    p.add_argument("--normalization_algorithm", type=str, default="library_size_normalization")
    p.add_argument("--library_size", type=float, default=25000)
    p.add_argument("--hic_smoothing", type=bool, default=True)
    p.add_argument("--smoothing_threshold", type=float, default=0.25)

    # Fine-tuning hyperparameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_step_size", type=int, default=30)
    p.add_argument("--lr_gamma", type=float, default=0.5)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--loss_scale", type=float, default=1.0)
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--early_stopping_patience", type=int, default=30,
                   help="0 to disable.")
    p.add_argument("--freeze_encoder", action="store_true", default=False,
                   help="Freeze ChIPSeqProcessor + GraphEncoder; train only the decoder.")
    p.add_argument("--seed", type=int, default=42)

    # TUI integration
    p.add_argument("--json_progress", action="store_true", default=False,
                   help="Emit one JSON metrics line per validation epoch (for the TUI).")
    p.add_argument("--validate_only", action="store_true", default=False,
                   help="Print a JSON path-validation report and exit without training.")

    return p.parse_args()


class JSONProgress(pl.Callback):
    # Emit one JSON line per validation epoch for the TUI to parse

    def on_validation_epoch_end(self, trainer, pl_module):
        import json
        m = trainer.callback_metrics

        def metric_float(key):
            v = m.get(key)
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        print(json.dumps({
            "epoch": int(trainer.current_epoch),
            "val_scc": metric_float("valid/SCC"),
            "val_gd": metric_float("valid/GD"),
            "val_ssim": metric_float("valid/SSIM"),
            "val_loss": metric_float("valid/loss"),
        }), flush=True)


def run_validate_only(args):
    # Print a JSON report on whether the required input paths exist, then return
    import json
    checks = [
        {"name": "checkpoint", "path": args.checkpoint or ""},
        {"name": "train_npz", "path": args.train_npz or ""},
        {"name": "val_npz", "path": args.val_npz or ""},
        {"name": "test_npz", "path": args.test_npz or ""},
        {"name": "output_dir", "path": args.output_dir or ""},
    ]
    for c in checks:
        if c["name"] == "output_dir":
            p = c["path"]
            c["exists"] = bool(p) and (
                os.path.isdir(p) or os.path.isdir(os.path.dirname(os.path.abspath(p)))
            )
        elif c["name"] in ("val_npz", "test_npz"):
            # Optional: blank is fine
            c["exists"] = (not c["path"]) or os.path.isfile(c["path"])
        else:
            c["exists"] = bool(c["path"]) and os.path.isfile(c["path"])
    print(json.dumps({"checks": checks, "ok": all(c["exists"] for c in checks)}))


def build_parameters(args):
    return {k: getattr(args, k) for k in [
        "experiment", "seed", "resolution", "library_size",
        "normalization_algorithm", "hic_smoothing", "smoothing_threshold",
        "bounds", "stride", "padding", "num_nodes", "remove_borders", "batch_size",
        "rna_seq", "use_bulk", "positional_encodings", "ctcf_motif", "cpg_motif",
        "node_features", "pos_encodings_dim",
        "conv1d_kernel_size", "encoder_hidden_embedding_size",
        "num_encoder_attn_blocks", "num_heads_encoder_attn_blocks",
        "num_graph_conv_blocks", "num_graph_encoder_blocks", "edge_dims",
        "num_decoder_residual_blocks", "width", "num_channels",
        "loss_scale", "epochs", "gradient_clip",
    ]} | {"gradient_clip_value": args.gradient_clip}


def load_datasets(args, params):
    train = GenomicDataset(args.train_npz, params)

    if args.val_npz:
        val = GenomicDataset(args.val_npz, params)
    else:
        warnings.warn("No --val_npz; holding out 10% of training data.", UserWarning)
        val_size = max(1, len(train) // 10)
        train, val = torch.utils.data.random_split(train, [len(train) - val_size, val_size])

    test = GenomicDataset(args.test_npz, params) if args.test_npz else val
    return train, val, test


def load_pretrained_model(checkpoint_path, params):
    model = scGrapHiC(params)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        warnings.warn(f"Missing keys: {missing}", UserWarning)
    if unexpected:
        warnings.warn(f"Unexpected keys: {unexpected}", UserWarning)
    return model


def configure_optimizer(model, args):
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if "chipseq_processor" in name or "graph_encoder" in name:
                param.requires_grad = False

    def new_configure_optimizers():
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    model.configure_optimizers = new_configure_optimizers


def main():
    args = parse_args()

    if args.validate_only:
        run_validate_only(args)
        return

    if args.mode == "raw":
        raise SystemExit(
            "raw mode is not supported: build the dataset first "
            "(python -m src.dataset_creator ...), then fine-tune with "
            "--mode npz --train_npz <built.npz>."
        )

    if not args.train_npz:
        raise SystemExit("--train_npz is required in npz mode.")
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required.")

    params = build_parameters(args)
    pl.seed_everything(args.seed)

    output_dir = os.path.abspath(args.output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    for d in [ckpt_dir, log_dir]:
        create_directory(d)

    train_ds, val_ds, test_ds = load_datasets(args, params)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = load_pretrained_model(args.checkpoint, params)
    configure_optimizer(model, args)

    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, filename="{epoch:03d}-{valid/SCC:.4f}",
                        monitor="valid/SCC", save_top_k=3, mode="max", save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor="valid/SCC", patience=args.early_stopping_patience, mode="max", verbose=True))
    if args.json_progress:
        callbacks.append(JSONProgress())

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_every,
        logger=TensorBoardLogger(log_dir, name=args.experiment),
        deterministic=True,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip,
        accelerator="auto",
        devices=1,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        trainer.test(model, test_loader, ckpt_path=best_ckpt)
    else:
        trainer.test(model, test_loader)

    final_path = os.path.join(output_dir, f"{args.experiment}_final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"Final model: {final_path}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()


"""
python finetune_trimmed.py \
    --freeze_encoder \
    --early_stopping_patience 5 \
    --train_npz /users/mliu237/data/mliu237/scHiCAR_brain_bug_update_0405/scHiCAR_brain/processed/scGrapHiC_train.npz \
    --test_npz  /users/mliu237/data/mliu237/scHiCAR_brain_bug_update_0405/scHiCAR_brain/processed/scGrapHiC_test.npz \
    --checkpoint /users/mliu237/data/mliu237/scHiCAR_brain_bug_update_0405/weights/epoch=499-step=43000.ckpt \
    --output_dir /users/mliu237/data/mliu237/scHiCAR_brain_bug_update_0405/weights/finetune \
    --experiment brain_finetune \
    --encoder_hidden_embedding_size 64 \
    --num_graph_conv_blocks 3 \
    --num_heads_encoder_attn_blocks 1 \
    --pos_encodings_dim 16 \
    --loss_scale 10 \
    --gradient_clip 0.5
"""