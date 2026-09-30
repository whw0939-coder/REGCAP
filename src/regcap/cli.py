"""Command-line entry points for the REGCAP reference pipeline."""

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from .data_loading import GlobalDataModule
from .model import VulDetectionModel
from .preprocessing import DatasetBuilder
from .rdp_schema import RDP_DIM, RDP_FEATURES
from .training import VulDetectionSystem


_DATASETS = {
    "ffmpeg_qemu": ("devign/c", "devign/js", "devign/Devign_vulnerable_lines.json", "devign/W2V/Devign-128-20.wordvectors"),
    "diversevul": ("DiverseVul/c", "DiverseVul/js", "DiverseVul/DiverseVul_vulnerable_lines.json", "DiverseVul/W2V/DiverseVul-128-20.wordvectors"),
    "primevul_train": ("PrimeVul/train_c", "PrimeVul/train_js", "PrimeVul/vulnerable_lines_train.json", "PrimeVul/w2v/primevul-128-20.wordvectors"),
    "primevul_valid": ("PrimeVul/valid_c", "PrimeVul/valid_js", "PrimeVul/vulnerable_lines_valid.json", "PrimeVul/w2v/primevul-128-20.wordvectors"),
    "primevul_test": ("PrimeVul/test_c", "PrimeVul/test_js", "PrimeVul/vulnerable_lines_test.json", "PrimeVul/w2v/primevul-128-20.wordvectors"),
}


def _paths(args):
    if args.data_root:
        defaults = tuple(Path(args.data_root) / part for part in _DATASETS[args.dataset])
    else:
        defaults = (None,) * 4
    paths = tuple(Path(value) if value else default for value, default in zip(
        (args.c_dir, args.js_dir, args.vul_lines, args.w2v), defaults
    ))
    if any(path is None for path in paths):
        raise ValueError("Provide --data-root or all four input paths")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return tuple(str(path) for path in paths)


def _model_args():
    return SimpleNamespace(input_dim=128, hidden_dim=128, gat_heads=4, ggnn_steps=6)


def _device(args):
    return "gpu" if args.gpu is not None and torch.cuda.is_available() else "cpu"


def _trainer(args, callbacks=None, max_epochs=None):
    accelerator = _device(args)
    devices = [args.gpu] if accelerator == "gpu" else 1
    return pl.Trainer(
        accelerator=accelerator, devices=devices, callbacks=callbacks or [],
        max_epochs=max_epochs or 1, default_root_dir=str(args.output_dir),
        enable_checkpointing=bool(callbacks), logger=False,
        enable_progress_bar=True,
    )


def _test_loader(path, batch_size, max_regions):
    with Path(path).open("rb") as stream:
        samples = pickle.load(stream)
    if not samples:
        raise ValueError("The processed input contains no samples")
    module = GlobalDataModule(data_path=str(path), batch_size=batch_size,
                              max_regions=max_regions, emb_dim=128,
                              save_lists_dir=None)
    module.global_test = samples
    return module.test_dataloader()


def _checkpoint(path, output_dir, tag, dump_scores):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    saved = checkpoint.get("hyper_parameters", {})
    if int(saved.get("attr_dim", RDP_DIM)) != RDP_DIM:
        raise ValueError("Checkpoint RDP dimension is not 31")
    try:
        model = VulDetectionSystem.load_from_checkpoint(
            str(path), map_location="cpu", model_args=_model_args(),
            attr_dim=RDP_DIM, pred_dump_dir=str(output_dir),
            pred_tag=tag, dump_region_scores=dump_scores,
        )
    except RuntimeError as exc:
        raise ValueError("Checkpoint architecture is incompatible with the released 31-dimensional, five-view REGCAP model") from exc
    return model


def preprocess(args):
    c_dir, js_dir, labels, w2v = _paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = DatasetBuilder(c_dir, js_dir, labels, w2v, seed=args.seed)
    builder.dataset_name = args.dataset
    builder.output_dir = str(args.output_dir)
    samples = builder.build_classify_dataset()
    if not samples:
        raise RuntimeError("Preprocessing produced no samples; inspect the error log")
    for sample in samples:
        if sample.region_attr.ndim != 2 or sample.region_attr.size(1) != RDP_DIM:
            raise ValueError("Preprocessing produced a noncanonical RDP vector")
        if tuple(sample.region_attr_names) != RDP_FEATURES:
            raise ValueError("Preprocessing produced a noncanonical feature order")
    target = args.output_dir / (args.dataset + "-31.pkl")
    if target.exists() and not args.overwrite:
        raise FileExistsError(target)
    builder.save_processed_data(samples, str(target))
    print(f"Wrote {len(samples)} samples to {target}")


def train(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.data:
        if any((args.train, args.valid, args.test)):
            raise ValueError("Use either --data or --train/--valid/--test")
    elif not all((args.train, args.valid, args.test)):
        raise ValueError("Supply --data or all of --train/--valid/--test")
    pl.seed_everything(args.seed, workers=True)
    module = GlobalDataModule(
        data_path=str(args.data) if args.data else None,
        train_data_path=str(args.train) if args.train else None,
        val_data_path=str(args.valid) if args.valid else None,
        test_data_path=str(args.test) if args.test else None,
        batch_size=args.batch_size, max_regions=args.max_regions, emb_dim=128,
        seed=args.seed, save_lists_dir=str(args.output_dir / "split_lists"),
    )
    module.prepare_data()
    module.setup(stage="fit")
    system = VulDetectionSystem(
        model_args=_model_args(), attr_dim=RDP_DIM, lr=args.learning_rate,
        lambda_reg=1.0, lambda_cls=1.0,
        use_dynamic_threshold=False,
        pred_dump_dir=str(args.output_dir / "predictions"), pred_tag="test",
        dump_region_scores=True,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(args.output_dir / "checkpoints"), monitor="val_f1",
        mode="max", save_top_k=1, save_last=True,
        filename="regcap-{epoch:02d}-{val_f1:.4f}",
    )
    trainer = _trainer(args, callbacks=[checkpoint], max_epochs=args.epochs)
    trainer.fit(system, train_dataloaders=module.train_dataloader(),
                val_dataloaders=module.val_dataloader())
    if checkpoint.best_model_path:
        trainer.test(system, dataloaders=module.test_dataloader(),
                     ckpt_path=checkpoint.best_model_path)
        print("Best checkpoint:", checkpoint.best_model_path)


def evaluate(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    system = _checkpoint(args.checkpoint, args.output_dir, "evaluation", True)
    trainer = _trainer(args)
    trainer.test(system, dataloaders=_test_loader(args.data, args.batch_size,
                                                  args.max_regions))
    print("Metrics and predictions:", args.output_dir)


def predict(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    system = _checkpoint(args.checkpoint, args.output_dir, "prediction", True)
    system.eval()
    device = torch.device("cuda", args.gpu) if _device(args) == "gpu" else torch.device("cpu")
    system.to(device)
    target = args.output_dir / "region_predictions.jsonl"
    with target.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for batch in _test_loader(args.data, args.batch_size, args.max_regions):
            batch = batch.to(device)
            output = system.model(batch)
            probabilities = torch.sigmoid(output["cls_logits"]).cpu()
            scores = torch.sigmoid(output["region_logits"]).cpu()
            masks = output["region_mask"].cpu()
            for index, name in enumerate(batch.file_name):
                record = {
                    "file": name,
                    "probability": float(probabilities[index]),
                    "prediction": int(probabilities[index] >= system.decision_threshold.cpu()),
                    "threshold": float(system.decision_threshold.cpu()),
                    "region_scores": scores[index][masks[index]].tolist(),
                }
                stream.write(json.dumps(record) + "\n")
    print("Wrote region predictions to", target)


def smoke(args):
    if RDP_DIM != 31 or len(RDP_FEATURES) != RDP_DIM:
        raise AssertionError("RDP schema is inconsistent")
    c_file, js_file = Path(args.c_file), Path(args.js_file)
    with Path(args.vul_lines).open(encoding="utf-8") as stream:
        labels = json.load(stream)
    item = labels.get(c_file.name, [])
    if isinstance(item, dict):
        item = item.get("vulnerable_lines", [])
    builder = DatasetBuilder(str(c_file.parent), str(js_file.parent),
                             str(args.vul_lines), str(args.w2v), seed=42)
    sample = builder.prepare_torch_graph(str(js_file), str(c_file), item)
    assert sample.region_attr.shape[-1] == RDP_DIM
    assert tuple(sample.region_attr_names) == RDP_FEATURES
    assert torch.isfinite(sample.region_score_gt).all()
    assert ((sample.region_score_gt >= 0) & (sample.region_score_gt <= 1)).all()
    module = GlobalDataModule(data_path="", batch_size=1, max_regions=17,
                              emb_dim=128, save_lists_dir=None)
    batch = module._build_cls_batch([sample], split="test")
    assert batch.region_attr.shape[-1] == RDP_DIM
    model = VulDetectionModel(hidden_dim=128, input_dim=128, attr_dim=RDP_DIM)
    model.eval()
    with torch.inference_mode():
        output = model(batch)
    gates = output["view_gate_weights"][output["region_mask"]]
    assert gates.ndim == 2 and gates.size(-1) == 5
    assert torch.allclose(gates.sum(dim=-1), torch.ones(gates.size(0)), atol=1e-5)
    probability = torch.sigmoid(output["cls_logits"])[0]
    assert torch.isfinite(probability)
    scores = torch.sigmoid(output["region_logits"])[0][output["region_mask"][0]]
    assert torch.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    print(json.dumps({"rdp_dim": RDP_DIM, "view_gates": 5, "regions": int(scores.numel()),
                      "probability": float(probability),
                      "region_scores": scores.tolist()}))


def main(argv=None):
    parser = argparse.ArgumentParser(description="REGCAP reference implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("preprocess", help="Build 31-feature graph samples")
    prep.add_argument("--dataset", choices=tuple(_DATASETS), required=True)
    prep.add_argument("--data-root", type=Path)
    prep.add_argument("--c-dir", type=Path)
    prep.add_argument("--js-dir", type=Path)
    prep.add_argument("--vul-lines", type=Path)
    prep.add_argument("--w2v", type=Path)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(run=preprocess)

    training = commands.add_parser("train", help="Train and evaluate REGCAP")
    training.add_argument("--data", type=Path)
    training.add_argument("--train", type=Path)
    training.add_argument("--valid", type=Path)
    training.add_argument("--test", type=Path)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--max-regions", type=int, default=17)
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--gpu", type=int)
    training.set_defaults(run=train)

    for name, action in (("evaluate", evaluate), ("predict", predict)):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--max-regions", type=int, default=17)
        command.add_argument("--gpu", type=int)
        command.set_defaults(run=action)

    test = commands.add_parser("smoke", help="Run one user-provided sample end to end")
    test.add_argument("--c-file", type=Path, required=True)
    test.add_argument("--js-file", type=Path, required=True)
    test.add_argument("--vul-lines", type=Path, required=True)
    test.add_argument("--w2v", type=Path, required=True)
    test.set_defaults(run=smoke)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
