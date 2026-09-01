from pathlib import Path
import sys, re, glob, warnings
import numpy as np
import pandas as pd
import cv2, torch, yaml
from PIL import Image
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from ultralytics import YOLO
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "second_level_results"
OUT.mkdir(exist_ok=True)

EXPERT_CSV = DATA / "tangent_sign_expert_annotation.csv"
N_FOLDS = 5
INCLUDE_AUGMENTED = False

SEG_CKPT_GLOB = str(ROOT / "segmentation" / "check" / "*fold_{fold}*.ckpt")
KEYPOINT_WEIGHT = "best.pt"
THRESHOLDS = np.linspace(0, 100, 1001)
CACHE = OUT / "continuous_scores.csv"
MAX_FPR = 0.10
STRATEGIES = ["max_threshold_at_fpr10", "max_f1", "no_false_negatives"]


def strip_aug(x): return re.sub(r"_aug$", "", str(x))
def case_id(fname): return strip_aug(Path(str(fname)).stem)
def is_aug(fname): return Path(str(fname)).stem.endswith("_aug")

def read_mask(path):
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)

def files_from_dataset(ds):
    return [ds.dataset.dataset[i] for i in ds.indices] if isinstance(ds, Subset) else list(ds.dataset)

def load_study_cases():
    """Load and validate the cohort defined by all fold CSVs."""

    fold_case_sets = []

    for fold in range(N_FOLDS):
        split_path = (
            DATA
            / f"split_labels_fold_{fold}.csv"
        )

        split_df = pd.read_csv(split_path)

        cases = {
            case_id(value)
            for value in split_df[
                "patient"
            ].astype(str)
        }

        fold_case_sets.append(cases)

    reference_cases = fold_case_sets[0]

    for fold, fold_cases in enumerate(
        fold_case_sets[1:],
        start=1,
    ):
        if fold_cases != reference_cases:
            raise ValueError(
                "The split CSVs do not contain "
                "the same study cohort. "
                f"Fold {fold} differs from fold 0."
            )

    return reference_cases

def line_y(line_mask):
    h, w = line_mask.shape
    y = np.full(w, h, dtype=int)
    for x in range(w):
        nz = np.where(line_mask[:, x] > 0)[0]
        if len(nz): y[x] = nz[0]
    return y

def percent_above_line(muscle, line_mask):
    muscle = (muscle > 0).astype(np.uint8)
    total = muscle.sum()
    if total == 0: return np.nan
    h, w = muscle.shape
    y = line_y(line_mask)
    above = np.zeros_like(muscle)
    for x in range(w):
        above[:min(max(y[x], 0), h), x] = muscle[:min(max(y[x], 0), h), x]
    return 100.0 * above.sum() / total

def make_line_from_points(p1, p2, shape):
    h, w = shape
    x1, y1 = p1; x2, y2 = p2
    line = np.zeros((h, w), dtype=np.uint8)
    if abs(x2 - x1) < 1e-6:
        x = int(round(x1))
        if 0 <= x < w: line[:, x] = 1
        return line
    m, b = (y2 - y1) / (x2 - x1), y1 - ((y2 - y1) / (x2 - x1)) * x1
    for x in range(w):
        y = int(round(m * x + b))
        if 0 <= y < h: line[y, x] = 1
    return line

def load_expert(study_cases):
    df = pd.read_csv(EXPERT_CSV, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    label_col = next(
        (
            c for c in [
                "tangent_sign_numeric",
                "expert_label_numeric",
                "label",
            ]
            if c in df.columns
        ),
        None,
    )

    if label_col is None:
        raise ValueError(
            "Need label column: tangent_sign_numeric, "
            "expert_label_numeric, or label"
        )

    if "image_file" not in df.columns:
        raise ValueError("Expert CSV needs image_file column")

    df["case_id"] = df["image_file"].apply(case_id)
    df["label"] = df[label_col].astype(int)

    df = df.loc[
        df["case_id"].isin(study_cases)
    ].copy()

    # Make sure duplicate case_ids do not have conflicting labels
    conflicting = (
        df.groupby("case_id")["label"]
        .nunique()
    )

    conflicting = conflicting[
        conflicting > 1
    ]

    if not conflicting.empty:
        raise ValueError(
            "Conflicting expert labels for: "
            + ", ".join(conflicting.index)
        )

    # One row per actual case
    df = df.drop_duplicates(
        subset="case_id"
    )

    return df[
        ["case_id", "image_file", "label"]
    ].reset_index(drop=True)

def import_segmentation():
    sys.path.insert(0, str(ROOT / "segmentation"))
    from src.dataset import ROIDataModule
    from src.models import SegmentationModel
    from utils.utils_preproc import z_score_normalize_batch, filter_segments, skeletonize_mask, fit_line
    return ROIDataModule, SegmentationModel, z_score_normalize_batch, filter_segments, skeletonize_mask, fit_line

def make_dm(ROIDataModule, fold, dm_args):
    try:
        return ROIDataModule(fold_index=fold, run_prepare=False, **dm_args)
    except TypeError:
        warnings.warn("ROIDataModule has no run_prepare=False; it may rerun preprocessing.")
        return ROIDataModule(fold_index=fold, **dm_args)

def find_ckpt(fold):
    hits = sorted(glob.glob(SEG_CKPT_GLOB.format(fold=fold)))
    if not hits: raise FileNotFoundError(f"No segmentation checkpoint for fold {fold}")
    return hits[-1]

def keep_file(fname, labels):
    cid = case_id(fname)
    return not (
        (is_aug(fname) and not INCLUDE_AUGMENTED)
        or cid not in labels
    )

def score_segmentation(labels):
    ROIDataModule, SegmentationModel, norm, filt, skel, fit_line = import_segmentation()
    cfg = yaml.safe_load(open(ROOT / "segmentation/config/seg_tangent_sign.yml"))["datamodule_args"]
    cfg["root_dir"] = str(DATA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for fold in range(N_FOLDS):
        print(f"[segmentation] fold {fold}")
        dm = make_dm(ROIDataModule, fold, cfg)

        print("SEG SPLIT FILE:", dm.split_labels_path)

        split_df = pd.read_csv(dm.split_labels_path)

        print(
            split_df["phase"].value_counts()
        )

        print(
            "Unique cases in split:",
            split_df["patient"].nunique()
        )
        print("SEG SPLIT FILE:", Path(dm.split_labels_path).resolve())

        model = SegmentationModel.load_from_checkpoint(find_ckpt(fold)).to(device).eval()
        model.set_data_module(dm); model.set_fold(fold)

        dm.setup(stage="fit")
        dm.train_dataset.dataset.transform = dm.test_transform
        if hasattr(model, "calibrate_threshold_from_loader"):
            model.calibrate_threshold_from_loader(dm.val_dataloader())
        splits = [("train", dm.train_dataset), ("val", dm.val_dataset)]
        dm.setup(stage="test")
        splits.append(("test", dm.test_dataset))
        supra_dir = Path(dm.data_dir) / "masks" / dm.img_type / "supraspinatus"

        for split, ds in splits:
            filenames = files_from_dataset(ds)
            loader = DataLoader(ds, batch_size=dm.batch_size, shuffle=False, num_workers=dm.num_workers)
            idx = 0
            with torch.no_grad():
                for batch in loader:
                    pred = torch.nan_to_num(model.model(norm(batch["image"].to(device))).sigmoid(), nan=0.0)
                    pred = (pred > float(model.threshold_pred)).float()
                    for j in range(pred.shape[0]):
                        fname, idx = filenames[idx], idx + 1
                        if not keep_file(fname, labels): continue
                        supra = supra_dir / fname
                        if not supra.exists():
                            print(f"missing supraspinatus mask: {fname}"); continue
                        line = fit_line(skel(filt(pred[j, 0].detach().cpu().numpy(), size_threshold=10, distance_threshold=100)))
                        if isinstance(line, torch.Tensor): line = line.detach().cpu().numpy()
                        muscle = read_mask(supra)
                        if muscle.shape != line.shape:
                            muscle = cv2.resize(muscle, (line.shape[1], line.shape[0]), interpolation=cv2.INTER_NEAREST)
                        rows.append(dict(model="segmentation", fold=fold, split=split, case_id=case_id(fname), image_file=fname, percent_above=percent_above_line(muscle, line)))
    return pd.DataFrame(rows)

def score_keypoints(labels):
    rows = []
    yolo_root = DATA / "keypoints/yolo_dataset_5fold"
    muscle_dir = DATA / "keypoints/original/muscle_segmentation"

    for fold in range(N_FOLDS):
        print(f"[keypoint] fold {fold}")
        weights = ROOT / f"keypoints/models_5fold/fold_{fold}/weights/{KEYPOINT_WEIGHT}"
        if not weights.exists(): raise FileNotFoundError(weights)
        model = YOLO(str(weights))

        for split in ["train", "val", "test"]:
            img_dir = yolo_root / f"fold_{fold}/images/{split}"
            if not img_dir.exists(): continue
            for img_path in sorted(img_dir.glob("*.jpg")):
                fname = img_path.name
                if not keep_file(fname, labels): continue
                muscle_path = muscle_dir / fname
                if not muscle_path.exists():
                    print(f"missing supraspinatus mask: {fname}"); continue

                muscle, pct, status = read_mask(muscle_path), 100.0, "no_prediction"
                res = model.predict(
                    Image.open(img_path).convert("RGB"),
                    conf=0.0,
                    max_det=1,
                    imgsz=512,
                    verbose=False,
                )
                xy = None if not res or res[0].keypoints is None else res[0].keypoints.xy
                if xy is not None and len(xy) > 0:
                    pts = sorted(xy[0].detach().cpu().numpy().tolist(), key=lambda p: p[1])[:2]
                    if len(pts) == 2:
                        pct, status = percent_above_line(muscle, make_line_from_points(pts[0], pts[1], muscle.shape)), "ok"
                rows.append(dict(model="keypoint", fold=fold, split=split, case_id=case_id(fname), image_file=fname, percent_above=pct, status=status))
    return pd.DataFrame(rows)

def cls_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    den = 2 * tp + fp + fn
    return {
        "n": len(y_true), "n_positive": int((y_true == 1).sum()), "n_negative": int((y_true == 0).sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "f1": 2 * tp / den if den else np.nan,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "sensitivity_recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
    }

def metrics_at(df, thr):
    m = cls_metrics(df["label"].astype(int).values, (df["percent_above"].astype(float).values <= thr).astype(int))
    m["threshold"] = float(thr)
    return m

def tune(df, strategy):
    candidates = [metrics_at(df, t) for t in THRESHOLDS]
    if strategy == "max_threshold_at_fpr10":
        valid = [c for c in candidates if not np.isnan(c["fpr"]) and c["fpr"] <= MAX_FPR]
        best = sorted(valid, key=lambda c: c["threshold"], reverse=True)[0] if valid else sorted(candidates, key=lambda c: (np.inf if np.isnan(c["fpr"]) else c["fpr"], -c["threshold"]))[0]
    elif strategy == "max_f1":
        valid = [c for c in candidates if not np.isnan(c["f1"])]
        if not valid: return {"threshold": 0.0, "constraint_satisfied": False}
        best = sorted(valid, key=lambda c: (-c["f1"], -c["sensitivity_recall"], c["threshold"]))[0]
    elif strategy == "no_false_negatives":
        valid = [c for c in candidates if c["fn"] == 0 and c["n_positive"] > 0]
        best = sorted(valid, key=lambda c: (c["threshold"], np.inf if np.isnan(c["fpr"]) else c["fpr"]))[0] if valid else sorted(candidates, key=lambda c: (c["fn"], np.inf if np.isnan(c["fpr"]) else c["fpr"], c["threshold"]))[0]
    else:
        raise ValueError(strategy)
    best["constraint_satisfied"] = bool(strategy != "max_threshold_at_fpr10" or [c for c in candidates if not np.isnan(c["fpr"]) and c["fpr"] <= MAX_FPR]) if strategy == "max_threshold_at_fpr10" else bool(strategy != "no_false_negatives" or (best["fn"] == 0 and best["n_positive"] > 0))
    return best

def can_calibrate(calib, strategy):
    n_pos, n_neg = int((calib["label"] == 1).sum()), int((calib["label"] == 0).sum())
    return not (calib.empty or
                (strategy == "max_threshold_at_fpr10" and n_neg == 0) or
                (strategy == "max_f1" and (n_pos == 0 or n_neg == 0)) or
                (strategy == "no_false_negatives" and n_pos == 0))

def evaluate(scores, expert, strategy):
    df = scores.merge(expert[["case_id", "label"]], on="case_id", how="inner").dropna(subset=["percent_above", "label"]).copy()
    preds, thresholds, fold_metrics = [], [], []

    for model_name in sorted(df["model"].unique()):
        for fold in range(N_FOLDS):
            g = df[(df["model"] == model_name) & (df["fold"] == fold)]
            calib, test = g[g["split"].isin(["train", "val"])], g[g["split"] == "test"].copy()
            if test.empty: continue
            if can_calibrate(calib, strategy):
                cal = tune(calib, strategy)
            else:
                warnings.warn(f"{model_name} fold {fold}: calibration not possible for {strategy}; using threshold 0")
                cal = {"threshold": 0.0, "constraint_satisfied": False}

            test["strategy"] = strategy
            test["threshold_percent_above"] = cal["threshold"]
            test["pred_label"] = (test["percent_above"].astype(float) <= cal["threshold"]).astype(int)
            preds.append(test)
            thresholds.append({
                "model": model_name, "fold": fold, "strategy": strategy,
                "threshold_percent_above": cal["threshold"], "constraint_satisfied": cal.get("constraint_satisfied", False),
                "n_calib": len(calib), "n_calib_positive": int((calib["label"] == 1).sum()), "n_calib_negative": int((calib["label"] == 0).sum()),
                "n_test": len(test), "n_test_positive": int((test["label"] == 1).sum()), "n_test_negative": int((test["label"] == 0).sum()),
                "calibration_f1": cal.get("f1", np.nan), "calibration_recall": cal.get("sensitivity_recall", np.nan),
                "calibration_fpr": cal.get("fpr", np.nan), "calibration_specificity": cal.get("specificity", np.nan),
            })
            m = cls_metrics(test["label"].astype(int).values, test["pred_label"].astype(int).values)
            m.update({"model": model_name, "fold": fold, "strategy": strategy})
            fold_metrics.append(m)

    pred_df, thr_df, met_df = pd.concat(preds, ignore_index=True), pd.DataFrame(thresholds), pd.DataFrame(fold_metrics)
    pooled = []
    for model_name, g in pred_df.groupby("model"):
        m = cls_metrics(g["label"].astype(int).values, g["pred_label"].astype(int).values)
        m.update({"model": model_name, "strategy": strategy})
        pooled.append(m)
    return pred_df, thr_df, met_df, pd.DataFrame(pooled)

def pretty_strategy_name(strategy):
    return {
        "max_threshold_at_fpr10": "Max threshold at FPR ≤ 10%",
        "max_f1": "Max F1",
        "no_false_negatives": "No false negatives",
        "max_recall_at_fpr10": "Max recall at FPR ≤ 10%",
    }.get(strategy, strategy)

def roc_input(df):
    return df["label"].astype(int).values, -df["percent_above"].astype(float).values

def operating_point_from_thresholds(merged, thresholds, model_name, splits):
    g = merged[(merged["model"] == model_name) & (merged["split"].isin(splits))].copy()
    g = g.merge(thresholds[["model", "fold", "threshold_percent_above"]], on=["model", "fold"], how="inner")
    if g.empty: return None
    pred = (g["percent_above"].astype(float) <= g["threshold_percent_above"].astype(float)).astype(int)
    m = cls_metrics(g["label"].astype(int).values, pred.values)
    return {"fpr": m["fpr"], "tpr": m["sensitivity_recall"], "f1": m["f1"], "accuracy": m["accuracy"], "precision": m["precision"], "specificity": m["specificity"], "n": m["n"], "n_positive": m["n_positive"], "n_negative": m["n_negative"], "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]}

def plot_roc_curves(scores, expert, strategy_outputs):
    merged = scores.merge(
        expert[["case_id", "label"]],
        on="case_id",
        how="inner",
    ).dropna(subset=["percent_above", "label"]).copy()

    roc_rows = []

    strategy_colors = {
        "max_threshold_at_fpr10": "tab:blue",
        "max_f1": "tab:orange",
        "no_false_negatives": "tab:green",
    }

    strategy_offsets = {
        "max_threshold_at_fpr10": (-0.008,  0.008),
        "max_f1":                  (0.008,  0.008),
        "no_false_negatives":      (0.000, -0.008),
    }

    model_names = sorted(merged["model"].unique())

    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(6.5 * len(model_names), 5.5),
        sharex=True,
        sharey=True,
    )

    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        g = merged[
            (merged["model"] == model_name)
            & (merged["split"].isin(["train", "val"]))
        ]

        if g.empty or g["label"].nunique() < 2:
            print(f"Skipping ROC for {model_name}: not both classes present")
            ax.set_axis_off()
            continue

        y_true, y_score = roc_input(g)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"ROC curve (AUC={auc_value:.3f})",
        )

        roc_rows.append({
            "model": model_name,
            "split_group": "train_val",
            "type": "roc_curve",
            "auc": auc_value,
            "n": len(g),
            "n_positive": int((g["label"] == 1).sum()),
            "n_negative": int((g["label"] == 0).sum()),
        })

        for strategy, output in strategy_outputs.items():
            op = operating_point_from_thresholds(
                merged=merged,
                thresholds=output["thresholds"],
                model_name=model_name,
                splits=["train", "val"],
            )

            if op is None or np.isnan(op["fpr"]) or np.isnan(op["tpr"]):
                continue

            dx, dy = strategy_offsets.get(strategy, (0.0, 0.0))

            ax.scatter(
                np.clip(op["fpr"] + dx, 0, 1),
                np.clip(op["tpr"] + dy, 0, 1),
                s=95,
                marker="o",
                color=strategy_colors.get(strategy),
                label=pretty_strategy_name(strategy),
            )

            roc_rows.append({
                "model": model_name,
                "split_group": "train_val",
                "type": "operating_point",
                "strategy": strategy,
                "strategy_label": pretty_strategy_name(strategy),
                "operating_fpr": op["fpr"],
                "operating_tpr_sensitivity": op["tpr"],
                "operating_f1": op["f1"],
                "operating_accuracy": op["accuracy"],
                "operating_precision": op["precision"],
                "operating_specificity": op["specificity"],
                "n": op["n"],
                "n_positive": op["n_positive"],
                "n_negative": op["n_negative"],
                "tp": op["tp"],
                "fp": op["fp"],
                "tn": op["tn"],
                "fn": op["fn"],
            })

        ax.plot([0, 1], [0, 1], "k--", alpha=0.25, label="Chance")

        roc_titles = {
            "segmentation": "U-Net Segmentation model",
            "keypoint": "YOLO Keypoint model",
        }

        ax.set_title(
            roc_titles.get(model_name, model_name),
            fontsize=13,
        )
        ax.set_xlabel("False-positive rate", fontsize=11)
        ax.set_ylabel("Sensitivity / true-positive rate", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(
            fontsize=11,
            loc="lower right",
            frameon=True,
            borderpad=0.8,
            labelspacing=0.5,
            handlelength=1.8,
            handletextpad=0.6,
        )

    fig.tight_layout()

    out_path = OUT / "roc_calibration_side_by_side.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved ROC plot: {out_path}")

    roc_df = pd.DataFrame(roc_rows)
    roc_df.to_csv(OUT / "roc_auc_and_operating_points.csv", index=False)
    print(f"Saved ROC operating-point table: {OUT / 'roc_auc_and_operating_points.csv'}")

def plot_confusion_matrices_old(strategy_outputs):
    rows = []
    for strategy, output in strategy_outputs.items():
        pred, models = output["predictions"].copy(), sorted(output["predictions"]["model"].unique())
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
        if len(models) == 1: axes = [axes]

        for ax, model_name in zip(axes, models):
            g = pred[pred["model"] == model_name]
            cm = confusion_matrix(g["label"].astype(int).values, g["pred_label"].astype(int).values, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            cm_percent = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0) * 100
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max() * 1.5, 1))
            ax.set(title=model_name, xlabel="Predicted label", ylabel="True label", xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]}\n{cm_percent[i, j]:.1f}%", ha="center", va="center", color="black")
            rows.append(dict(strategy=strategy, strategy_label=pretty_strategy_name(strategy), model=model_name, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(cm.sum()), n_negative=int(cm[0].sum()), n_positive=int(cm[1].sum())))

        fig.suptitle(f"Confusion matrix: {pretty_strategy_name(strategy)}")
        fig.tight_layout()
        out_path = OUT / f"confusion_matrix_{strategy}.png"
        fig.savefig(out_path, dpi=300); plt.close(fig)
        print(f"Saved confusion matrix: {out_path}")

    pd.DataFrame(rows).to_csv(OUT / "confusion_matrices.csv", index=False)
    print(f"Saved confusion matrix table: {OUT / 'confusion_matrices.csv'}")

def plot_confusion_matrices(strategy_outputs):
    rows = []

    strategies = list(strategy_outputs.keys())
    models = sorted(
        set().union(
            *[
                set(output["predictions"]["model"].unique())
                for output in strategy_outputs.values()
            ]
        )
    )

    n_strategies = len(strategies)
    n_models = len(models)
    n_cols = n_strategies * n_models

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(4.2 * n_cols, 4.8),
        squeeze=False,
    )
    axes = axes[0]

    GROUP_TITLE_FS = 16
    SUBPLOT_TITLE_FS = 14
    AXIS_LABEL_FS = 14
    TICK_LABEL_FS = 12
    CELL_TEXT_FS = 14

    col_idx = 0

    for strategy_idx, strategy in enumerate(strategies):
        pred = strategy_outputs[strategy]["predictions"].copy()
        strategy_label = pretty_strategy_name(strategy)

        for model_name in models:
            ax = axes[col_idx]
            col_idx += 1

            g = pred[pred["model"] == model_name].copy()

            if g.empty:
                ax.set_axis_off()
                continue

            y_true = g["label"].astype(int).values
            y_pred = g["pred_label"].astype(int).values

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            row_sums = cm.sum(axis=1, keepdims=True)
            cm_percent = np.divide(
                cm,
                row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=row_sums != 0,
            ) * 100

            ax.imshow(
                cm,
                cmap="Blues",
                vmin=0,
                vmax=max(cm.max() * 2.0, 1),
            )

            # Only the model name goes above each individual matrix.
            ax.set_title(model_name, fontsize=SUBPLOT_TITLE_FS)

            ax.set_xlabel("Predicted label", fontsize=AXIS_LABEL_FS)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Negative", "Positive"], fontsize=TICK_LABEL_FS)

            ax.set_yticks([0, 1])

            # Only the first strategy group keeps the true-label axis text.
            if strategy_idx == 0:
                ax.set_ylabel("True label", fontsize=AXIS_LABEL_FS)
                ax.set_yticklabels(["Negative", "Positive"], fontsize=TICK_LABEL_FS)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]}\n{cm_percent[i, j]:.1f}%",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=CELL_TEXT_FS,
                    )

            rows.append({
                "strategy": strategy,
                "strategy_label": strategy_label,
                "model": model_name,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "n": int(cm.sum()),
                "n_negative": int(cm[0].sum()),
                "n_positive": int(cm[1].sum()),
            })

    # Leave room at the top for the centered strategy titles.
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # Add one centered title above each strategy group.
    for strategy_idx, strategy in enumerate(strategies):
        strategy_label = pretty_strategy_name(strategy)

        start = strategy_idx * n_models
        end = start + n_models
        group_axes = axes[start:end]

        left = group_axes[0].get_position().x0
        right = group_axes[-1].get_position().x1
        x = (left + right) / 2

        fig.text(
            x,
            0.94,
            strategy_label,
            ha="center",
            va="center",
            fontsize=GROUP_TITLE_FS,
            fontweight="bold",
        )

    # Add subtle vertical delimiter lines between strategy groups.
    for boundary in range(1, n_strategies):
        left_ax = axes[boundary * n_models - 1]
        right_ax = axes[boundary * n_models]

        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()

        x = (left_pos.x1 + right_pos.x0) / 2

        fig.add_artist(
            plt.Line2D(
                [x, x],
                [0.10, 0.90],
                transform=fig.transFigure,
                color="0.75",
                linewidth=1.0,
            )
        )

    out_path = OUT / "confusion_matrices_all_strategies.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved combined confusion matrix plot: {out_path}")

    cm_df = pd.DataFrame(rows)
    cm_df.to_csv(OUT / "confusion_matrices.csv", index=False)
    print(f"Saved confusion matrix table: {OUT / 'confusion_matrices.csv'}")

def main():
    study_cases = load_study_cases()
    expert = load_expert(study_cases)
    labels = set(expert["case_id"])
    print(f"Expert cases: {len(labels)}")

    if CACHE.exists():
        print(f"Loading cached scores: {CACHE}")
        scores = pd.read_csv(CACHE)
    else:
        scores = pd.concat([score_segmentation(labels), score_keypoints(labels)], ignore_index=True)
        scores.to_csv(CACHE, index=False)
        print(f"Saved cached scores: {CACHE}")

    strategy_outputs = {}
    for strategy in STRATEGIES:
        print(f"\nStrategy: {strategy}")
        pred, thr, fold_met, pooled = evaluate(scores, expert, strategy)
        pred.to_csv(OUT / f"test_predictions_{strategy}.csv", index=False)
        thr.to_csv(OUT / f"thresholds_by_fold_{strategy}.csv", index=False)
        fold_met.to_csv(OUT / f"metrics_by_fold_{strategy}.csv", index=False)
        pooled.to_csv(OUT / f"pooled_test_metrics_{strategy}.csv", index=False)
        strategy_outputs[strategy] = {"predictions": pred, "thresholds": thr, "fold_metrics": fold_met, "pooled": pooled}
        print("\nPooled test metrics:"); print(pooled)
        print("\nThreshold summary:"); print(thr.groupby("model")["threshold_percent_above"].agg(["mean", "std", "median", "min", "max"]))

    plot_roc_curves(scores, expert, strategy_outputs)
    plot_confusion_matrices(strategy_outputs)

if __name__ == "__main__":
    main()
