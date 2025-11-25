import argparse
import json
import os
from collections import defaultdict
from statistics import median
from typing import Dict, List, Tuple, Set

from labels import label_is_pii


def load_gold(path: str):
    gold = {}
    texts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            uid = obj["id"]
            texts[uid] = obj["text"]
            spans = []
            for e in obj.get("entities", []):
                spans.append((e["start"], e["end"], e["label"]))
            gold[uid] = spans
    return gold, texts


def load_pred(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pred = {}
    for uid, ents in obj.items():
        spans = []
        for e in ents:
            spans.append((e["start"], e["end"], e["label"]))
        pred[uid] = spans
    return pred


def compute_counts(gold: Dict[str, List[Tuple[int, int, str]]], pred: Dict[str, List[Tuple[int, int, str]]]):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    labels: Set[str] = set()
    for spans in gold.values():
        for _, _, lab in spans:
            labels.add(lab)
    for uid in gold.keys():
        g_spans = set(gold.get(uid, []))
        p_spans = set(pred.get(uid, []))
        for span in p_spans:
            if span in g_spans:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1
        for span in g_spans:
            if span not in p_spans:
                fn[span[2]] += 1
    return tp, fp, fn, labels


def prf(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def aggregate_micro(tp_dict, fp_dict, fn_dict):
    tp_total = sum(tp_dict.values())
    fp_total = sum(fp_dict.values())
    fn_total = sum(fn_dict.values())
    p, r, f1 = prf(tp_total, fp_total, fn_total)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp_total, "fp": fp_total, "fn": fn_total}


def group_pii(gold, pred):
    pii_tp = pii_fp = pii_fn = 0
    non_tp = non_fp = non_fn = 0
    for uid in gold.keys():
        g_spans = gold.get(uid, [])
        p_spans = pred.get(uid, [])
        g_pii = set((s, e, lab) for s, e, lab in g_spans if label_is_pii(lab))
        g_non = set((s, e, lab) for s, e, lab in g_spans if not label_is_pii(lab))
        p_pii = set((s, e, lab) for s, e, lab in p_spans if label_is_pii(lab))
        p_non = set((s, e, lab) for s, e, lab in p_spans if not label_is_pii(lab))
        for span in p_pii:
            if span in g_pii:
                pii_tp += 1
            else:
                pii_fp += 1
        for span in g_pii:
            if span not in p_pii:
                pii_fn += 1
        for span in p_non:
            if span in g_non:
                non_tp += 1
            else:
                non_fp += 1
        for span in g_non:
            if span not in p_non:
                non_fn += 1
    pii = prf(pii_tp, pii_fp, pii_fn)
    non = prf(non_tp, non_fp, non_fn)
    return {
        "pii": {"precision": pii[0], "recall": pii[1], "f1": pii[2], "tp": pii_tp, "fp": pii_fp, "fn": pii_fn},
        "non_pii": {"precision": non[0], "recall": non[1], "f1": non[2], "tp": non_tp, "fp": non_fp, "fn": non_fn},
    }


def span_length_stats(pred: Dict[str, List[Tuple[int, int, str]]]):
    lengths = []
    for spans in pred.values():
        for s, e, _ in spans:
            if e > s:
                lengths.append(e - s)
    if not lengths:
        return {"count": 0, "median": 0, "p95": 0, "mean": 0}
    lengths_sorted = sorted(lengths)
    p95 = lengths_sorted[int(0.95 * len(lengths_sorted)) - 1] if len(lengths_sorted) >= 1 else 0
    return {
        "count": len(lengths),
        "median": median(lengths_sorted),
        "p95": p95,
        "mean": sum(lengths_sorted) / len(lengths_sorted),
    }


def validation_drops(raw_pred, validated_pred):
    dropped = defaultdict(int)
    for uid in raw_pred.keys():
        raw_set = set(raw_pred.get(uid, []))
        val_set = set(validated_pred.get(uid, []))
        for span in raw_set - val_set:
            dropped[span[2]] += 1
    return dropped


def collect_false_positives(gold, pred, texts, labels_of_interest, limit_per_label=10):
    fps = defaultdict(list)
    for uid in gold.keys():
        g_set = set(gold.get(uid, []))
        for s, e, lab in pred.get(uid, []):
            if lab in labels_of_interest and (s, e, lab) not in g_set:
                if len(fps[lab]) < limit_per_label:
                    txt = texts.get(uid, "")
                    span_text = txt[s:e]
                    fps[lab].append({"start": s, "end": e, "text": span_text})
    return fps


def main():
    ap = argparse.ArgumentParser(description="Extended evaluation for PII NER with additional diagnostics.")
    ap.add_argument("--gold", required=True, help="Gold JSONL with entities.")
    ap.add_argument("--pred", required=True, help="Validated prediction JSON file.")
    ap.add_argument("--pred_raw", help="Raw prediction JSON file (before validation) to measure dropped spans.")
    ap.add_argument("--out_json", default="out/extended_metrics.json", help="Path to write extended metrics JSON.")
    ap.add_argument("--show_table", action="store_true", help="Print per-label table.")
    ap.add_argument("--false_positive_labels", nargs="*", default=["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"], help="Labels to list sample false positives for.")
    ap.add_argument("--fp_limit", type=int, default=10)
    args = ap.parse_args()

    gold, texts = load_gold(args.gold)
    pred_valid = load_pred(args.pred)
    pred_raw = load_pred(args.pred_raw) if args.pred_raw else None

    tp, fp, fn, labels = compute_counts(gold, pred_valid)
    per_label = {}
    macro_f1_sum = 0.0
    macro_count = 0
    for lab in sorted(labels):
        p, r, f1 = prf(tp[lab], fp[lab], fn[lab])
        per_label[lab] = {
            "tp": tp[lab], "fp": fp[lab], "fn": fn[lab], "precision": p, "recall": r, "f1": f1
        }
        macro_f1_sum += f1
        macro_count += 1
    macro_f1 = macro_f1_sum / max(1, macro_count)

    micro = aggregate_micro(tp, fp, fn)
    pii_group = group_pii(gold, pred_valid)
    span_stats = span_length_stats(pred_valid)

    dropped = {}
    if pred_raw:
        dropped_counts = validation_drops(pred_raw, pred_valid)
        dropped = {lab: dropped_counts.get(lab, 0) for lab in sorted(labels)}

    fps = collect_false_positives(gold, pred_valid, texts, set(args.false_positive_labels), limit_per_label=args.fp_limit)

    out_obj = {
        "dataset": os.path.basename(args.gold),
        "macro_f1": macro_f1,
        "micro": micro,
        "per_label": per_label,
        "pii_group": pii_group,
        "span_length": span_stats,
        "validation": {"dropped_counts": dropped, "raw_file": args.pred_raw, "validated_file": args.pred},
        "false_positives": fps,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    if args.show_table:
        print("Per-label metrics (validated predictions):")
        print(f"{'LABEL':15s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}")
        for lab in sorted(labels):
            m = per_label[lab]
            print(f"{lab:15s} {m['tp']:4d} {m['fp']:4d} {m['fn']:4d} {m['precision']:.3f} {m['recall']:.3f} {m['f1']:.3f}")
        print(f"\nMacro-F1: {macro_f1:.3f}")
        print("Micro:", micro)
        print("PII group:", pii_group["pii"])
        print("Non-PII group:", pii_group["non_pii"])
        if pred_raw:
            print("Dropped spans by validation:")
            for lab, cnt in dropped.items():
                if cnt > 0:
                    print(f"  {lab}: {cnt}")
        print("Span length stats:", span_stats)
        print("False positives samples:")
        for lab, samples in fps.items():
            print(f"  {lab}:")
            for s in samples:
                print(f"    [{s['start']},{s['end']}] '{s['text']}'")

    print(f"Extended metrics written to {args.out_json}")


if __name__ == "__main__":
    main()
