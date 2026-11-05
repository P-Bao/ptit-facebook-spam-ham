"""pipeline/modern/train.py
Fine-tune viBERT, lựa chọn bằng validation rồi đánh giá trên test cố định.

Đầu vào: v7, cấu hình JSON và kết quả S3 đã kiểm chứng.
Xử lý: smoke → hai loss trên fit/validation → chọn epoch/loss → fit toàn bộ train.
Đầu ra: model/tokenizer, lịch sử học, chỉ số test và dự đoán theo ID.
Liên quan: modern.data; modern.predict; notebook 03; không thay đổi baseline SVM.
"""

import argparse
import gc
import json
import platform
import importlib.metadata as metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix, precision_recall_curve)
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

from pipeline.modern.data import prepare_data, class_weights, split_summary
from pipeline.modern.predict import VibertPredictor


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class CommentDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.tokens = tokenizer(frame.model_text.tolist(), truncation=True, max_length=max_length)
        self.labels = frame.target.to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{k: torch.tensor(v[index]) for k, v in self.tokens.items()},
                "labels": torch.tensor(self.labels[index], dtype=torch.long)}


class ClassificationTrainer(Trainer):
    def __init__(self, *args, loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        weights = None if self.loss_weights is None else torch.as_tensor(
            self.loss_weights, dtype=outputs.logits.dtype, device=outputs.logits.device)
        loss = CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def probabilities(logits):
    logits = logits[0] if isinstance(logits, tuple) else logits
    return torch.softmax(torch.as_tensor(logits).float(), dim=-1)[:, 1].numpy()


def measure(labels, scores):
    predicted = np.asarray(scores) >= 0.5
    return {"accuracy": accuracy_score(labels, predicted),
            "precision": precision_score(labels, predicted, zero_division=0),
            "recall": recall_score(labels, predicted, zero_division=0),
            "f1": f1_score(labels, predicted, zero_division=0),
            "ap": average_precision_score(labels, scores)}


def compute_metrics(evaluation):
    return measure(evaluation.label_ids, probabilities(evaluation.predictions))


def load_pretrained(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=config["revision"],
                                              do_lower_case=False, token=False)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], revision=config["revision"], num_labels=2,
        id2label={0: "ham", 1: "spam"}, label2id={"ham": 0, "spam": 1},
        output_loading_info=True, token=False)
    missing_encoder = [key for key in info["missing_keys"] if key.startswith("bert.") and "pooler" not in key]
    if missing_encoder:
        raise ValueError(f"Thiếu trọng số encoder viBERT: {missing_encoder[:5]}")
    return tokenizer, model, info


def make_arguments(directory, config, epochs, validation):
    return TrainingArguments(
        output_dir=str(directory), learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"], warmup_ratio=config["warmup_ratio"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=epochs, eval_strategy="epoch" if validation else "no",
        save_strategy="epoch", save_total_limit=2, load_best_model_at_end=validation,
        metric_for_best_model="f1" if validation else None, greater_is_better=True,
        seed=config["seed"], data_seed=config["seed"], report_to="none",
        fp16=torch.cuda.is_available(), bf16=False, optim="adamw_torch",
        logging_steps=50, dataloader_num_workers=0, save_safetensors=True, disable_tqdm=True)


def fit_variant(data, config, directory, variant, epochs, validation):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"config":config, "variant":variant, "epochs":epochs,
                "validation":validation, "train_ids":data.loc[
                    data.development_split.eq("fit") if validation else data.outer_split.eq("train"), "id"].tolist()}
    spec = directory / "training_spec.json"
    if spec.exists() and json.loads(spec.read_text(encoding="utf-8")) != identity:
        raise ValueError("Cấu hình/ID đã thay đổi; dùng thư mục output mới để tránh resume nhầm.")
    save_json(spec, identity)
    set_seed(config["seed"])
    tokenizer, model, loading = load_pretrained(config)
    fit = data[data.development_split.eq("fit")] if validation else data[data.outer_split.eq("train")]
    valid = data[data.development_split.eq("validation")] if validation else None
    weights = class_weights(fit.target.to_numpy()).tolist() if variant == "balanced" else None
    trainer = ClassificationTrainer(model=model, args=make_arguments(directory,config,epochs,validation),
        train_dataset=CommentDataset(fit,tokenizer,config["max_length"]),
        eval_dataset=CommentDataset(valid,tokenizer,config["max_length"]) if validation else None,
        data_collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8),
        processing_class=tokenizer, compute_metrics=compute_metrics, loss_weights=weights)
    checkpoint = get_last_checkpoint(str(directory))
    trainer.train(resume_from_checkpoint=checkpoint)
    pd.DataFrame(trainer.state.log_history).to_csv(directory / "training_history.csv",index=False)
    save_json(directory / "loading_info.json", loading)
    save_json(directory / "class_weights.json", {"ham":weights[0], "spam":weights[1]} if weights else None)
    return trainer, tokenizer


def smoke(data, config, output):
    """Kiểm tra forward/backward thực với viBERT; không báo chỉ số như thực nghiệm đầy đủ."""
    set_seed(config["seed"])
    tokenizer, model, info = load_pretrained(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    sample = data[data.development_split.eq("fit")].groupby("label").head(1)
    inputs = tokenizer(sample.model_text.tolist(), padding=True, truncation=True,
                       max_length=config["max_length"],return_tensors="pt").to(device)
    labels = torch.tensor(sample.target.tolist(), device=device)
    loss = model(**inputs, labels=labels).loss
    loss.backward()
    if not torch.isfinite(loss):
        raise ValueError("Smoke loss không hữu hạn.")
    result = {"status":"passed", "scope":"real checkpoint forward/backward, not full training metrics",
              "device":device,"examples":len(sample),"loss":float(loss.detach().cpu()),
              "missing_keys":info["missing_keys"],"tokenizer_lowercase":tokenizer.do_lower_case}
    save_json(output / "smoke.json", result)
    del model
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def select(data, config, output):
    rows = []
    for variant in config["variants"]:
        save_json(output / "state.json", {"stage":"selection", "variant":variant})
        trainer, tokenizer = fit_variant(data,config,output / "selection" / variant,
                                          variant,config["epochs"],True)
        history = [row for row in trainer.state.log_history if "eval_f1" in row]
        chosen = max(history, key=lambda row:row["eval_f1"])
        rows.append({"variant":variant,"epoch":int(round(chosen["epoch"])),
                     **{key.removeprefix("eval_"):float(value) for key,value in chosen.items()
                        if key in {"eval_accuracy","eval_precision","eval_recall","eval_f1","eval_ap","eval_loss"}}})
        pd.DataFrame(rows).to_csv(output / "validation_metrics.csv",index=False)
        del trainer, tokenizer
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    selected = max(rows, key=lambda row:row["f1"])
    save_json(output / "selected.json", selected)
    return selected


def finalize(data, config, output, baseline_predictions):
    selected = json.loads((output / "selected.json").read_text(encoding="utf-8"))
    save_json(output / "state.json", {"stage":"final_train", "selected":selected})
    trainer, tokenizer = fit_variant(data,config,output / "final_training",selected["variant"],selected["epoch"],False)
    model_dir = output / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    save_json(model_dir / "inference_config.json", {"max_length":config["max_length"], "threshold":0.5,
                                                  "fp16":torch.cuda.is_available(),
                                                  "normalization":"NFC + HTML unescape + whitespace; preserve case"})
    test = data[data.outer_split.eq("test")].copy()
    result = trainer.predict(CommentDataset(test,tokenizer,config["max_length"]))
    scores = probabilities(result.predictions)
    metrics = measure(test.target.to_numpy(),scores)
    predictions = test[["id","text","label"]].copy()
    predictions["prediction"] = np.where(scores >= 0.5,"spam","ham")
    predictions["probability_spam"] = scores
    predictions.to_csv(output / "test_predictions.csv",index=False,encoding="utf-8-sig")
    row = {"Mã":"M1","Mô hình":"viBERT","Chiến lược":selected["variant"],
           **{name:metrics[key] for key,name in [("accuracy","Accuracy"),("precision","Precision"),
               ("recall","Recall"),("f1","F1"),("ap","AP")]}}
    pd.DataFrame([row]).to_csv(output / "metrics.csv",index=False,encoding="utf-8-sig")
    # Giải phóng Trainer trước khi nạp bundle để kiểm tra suy luận độc lập.
    del trainer, tokenizer, result
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    loaded = VibertPredictor(model_dir)
    check = loaded.predict(test.text.tolist(),batch_size=config["eval_batch_size"])
    assert check.du_doan.tolist() == predictions.prediction.tolist()
    np.testing.assert_allclose(check.probability_spam,scores,atol=1e-5,rtol=1e-5)
    comparison = compare_with_svm(predictions,baseline_predictions,row,output)
    draw_figures(test.target.to_numpy(),scores,comparison,output)
    save_json(output / "verification.json", {"status":"completed","test_rows":len(test),
              "test_spam":int(test.target.sum()),"reload_predictions_match":True,
              "selection_uses_validation_only":True,"threshold":0.5,"selected":selected,
              "metrics":metrics,"device":loaded.device})
    save_json(output / "state.json", {"stage":"completed", "selected":selected})
    return row


def compare_with_svm(predictions, baseline_path, modern_row, output):
    baseline = pd.read_csv(baseline_path,keep_default_na=False,dtype={"id":str})
    if not baseline.id.is_unique or set(baseline.id) != set(predictions.id):
        raise ValueError("S3 và viBERT phải có đúng cùng ID test, không trùng ID.")
    baseline = baseline.set_index("id").loc[predictions.id]
    assert baseline["Thực tế"].tolist() == predictions.label.tolist()
    y = predictions.label.eq("spam").astype(int).to_numpy()
    pred = baseline["Dự đoán"].eq("spam").astype(int).to_numpy()
    svm = {"Mã":"S3","Mô hình":"TF-IDF + SVM","Chiến lược":"Điều chỉnh ngưỡng",
           "Accuracy":accuracy_score(y,pred),"Precision":precision_score(y,pred),
           "Recall":recall_score(y,pred),"F1":f1_score(y,pred),
           "AP":average_precision_score(y,baseline["Điểm mô hình"])}
    frame = pd.DataFrame([svm,modern_row])
    frame.to_csv(output / "comparison_with_svm.csv",index=False,encoding="utf-8-sig")
    return {"table":frame,"svm_scores":baseline["Điểm mô hình"].to_numpy()}


def draw_figures(y, scores, comparison, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    matrix = confusion_matrix(y,scores >= 0.5,labels=[0,1])
    fig,ax = plt.subplots(figsize=(5,4))
    ax.imshow(matrix,cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",
                    color="white" if matrix[i,j] > matrix.max()/2 else "#111827")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["ham","spam"],yticklabels=["ham","spam"],
           xlabel="Dự đoán",ylabel="Thực tế",title="viBERT — ma trận nhầm lẫn")
    fig.tight_layout();fig.savefig(figures / "confusion_vibert.png",dpi=140);plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,5))
    for name,values in [("S3 — TF-IDF + SVM",comparison["svm_scores"]),("M1 — viBERT",scores)]:
        precision,recall,_ = precision_recall_curve(y,values)
        ax.plot(recall,precision,label=f"{name} (AP={average_precision_score(y,values):.3f})")
    ax.set(xlabel="Recall spam",ylabel="Precision spam",title="Precision–Recall trên cùng test")
    ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(figures / "pr_comparison.png",dpi=140);plt.close(fig)


def run_experiment(csv_path, output, baseline_predictions, config_path, stage="all"):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config["threshold"] != 0.5:
        raise ValueError("Thực nghiệm hiện tại dùng ngưỡng 0,5 cố định.")
    if stage != "smoke" and not torch.cuda.is_available():
        raise RuntimeError("Huấn luyện đầy đủ yêu cầu GPU Colab; CPU chỉ dùng kiểm tra smoke.")
    data = prepare_data(csv_path,config["validation_fold"])
    data[["id","label","outer_split","development_split"]].to_csv(output / "split_ids.csv",index=False)
    split_path = Path(csv_path).parent.parent / "processed/vibert_split.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    data[["id","label","outer_split","development_split"]].to_csv(split_path,index=False)
    split_summary(data).to_csv(output / "split_summary.csv")
    save_json(output / "run_config.json", {"config":config,"python":platform.python_version(),
        "torch":torch.__version__,"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "packages":{name:metadata.version(name) for name in ["transformers","accelerate","numpy","pandas","scipy","scikit-learn"]}})
    if stage in {"smoke","all"}:smoke(data,config,output)
    if stage in {"select","all"}:select(data,config,output)
    if stage in {"finalize","all"}:return finalize(data,config,output,baseline_predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",type=Path,default=Path("data/final/labeled_v7.csv"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/modern/vibert"))
    parser.add_argument("--baseline-predictions",type=Path,default=Path("artifacts/spamham/test_predictions.csv"))
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("vibert_config.json"))
    parser.add_argument("--stage",choices=["smoke","select","finalize","all"],default="all")
    args=parser.parse_args()
    run_experiment(args.csv,args.output,args.baseline_predictions,args.config,args.stage)


if __name__ == "__main__":main()
