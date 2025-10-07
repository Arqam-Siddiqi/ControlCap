import json
import os
import argparse
from typing import Set, Tuple, Dict, Any, List

def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def collect_predicted_ids(result_json: Dict[str, Any]) -> Tuple[Set[int], Set[int]]:

    img_ids = set()
    ann_ids = set()
    for ann in result_json.get("annotations", []):
        pred_res = ann.get("extra_info", {}).get("pred_result", None)
        if pred_res is not None:
            img_ids.add(ann["image_id"])
            ann_ids.add(ann["id"])
    return img_ids, ann_ids

def subset_ids_ordered(images: List[Dict[str, Any]], keep: Set[int], limit: int) -> List[int]:
    ordered = [im["id"] for im in images if im["id"] in keep]
    if limit > 0:
        ordered = ordered[:limit]
    return ordered

def build_subset_gt(orig_gt_path: str,
                    result_json: Dict[str, Any],
                    keep_image_ids: List[int]) -> Dict[str, Any]:
    gt_full = load_json(orig_gt_path)
    keep_set = set(keep_image_ids)
    subset = {
        "dataset": gt_full.get("dataset", {}),
        "images": [im for im in gt_full["images"] if im["id"] in keep_set],
        "annotations": [ann for ann in gt_full["annotations"] if ann["image_id"] in keep_set],
    }
    return subset

def build_pruned_result(result_json: Dict[str, Any],
                        keep_image_ids: List[int]) -> Dict[str, Any]:
    keep_set = set(keep_image_ids)
    pruned_anns = []
    for ann in result_json.get("annotations", []):
        if (ann["image_id"] in keep_set and
            ann.get("extra_info", {}).get("pred_result") is not None):
            pruned_anns.append(ann)
    pruned = {
        "dataset": result_json.get("dataset", {}),
        "images": [im for im in result_json.get("images", []) if im["id"] in keep_set],
        "annotations": pruned_anns
    }
    return pruned

def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"[INFO] Wrote {path} (images={len(obj.get('images', []))}, anns={len(obj.get('annotations', []))})")

def main():
    ap = argparse.ArgumentParser("Build subset GT & result for partial predictions")
    ap.add_argument("--res-path", required=True,
                    help="Path to merged val.json (partial predictions)")
    ap.add_argument("--orig-gt", required=True,
                    help="Path to original full GT test.json used by evaluator")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for subset_gt.json & subset_result.json")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Optional limit on number of predicted images to keep (in original order). -1 keeps all.")
    ap.add_argument("--dry-run", action="store_true", help="Only print stats; do not write files.")
    args = ap.parse_args()

    result_json = load_json(args.res_path)
    img_ids_with_preds, ann_ids_with_preds = collect_predicted_ids(result_json)

    if not img_ids_with_preds:
        raise ValueError("No predictions (extra_info.pred_result) found in provided result file.")

    print(f"[INFO] Found {len(img_ids_with_preds)} images and {len(ann_ids_with_preds)} predicted annotations.")

    # Maintain original ordering (using result_json images list)
    ordered_keep_ids = subset_ids_ordered(result_json.get("images", []),
                                          img_ids_with_preds,
                                          args.limit)

    if args.limit > 0:
        print(f"[INFO] Limiting to first {len(ordered_keep_ids)} predicted images (requested {args.limit}).")

    subset_gt = build_subset_gt(args.orig_gt, result_json, ordered_keep_ids)
    pruned_res = build_pruned_result(result_json, ordered_keep_ids)

    # Basic sanity checks
    gt_img_ids = {im["id"] for im in subset_gt["images"]}
    res_img_ids = {im["id"] for im in pruned_res["images"]}
    if gt_img_ids != res_img_ids:
        missing_in_gt = res_img_ids - gt_img_ids
        missing_in_res = gt_img_ids - res_img_ids
        raise AssertionError(f"Image id mismatch between subset GT and pruned result. "
                             f"Missing in GT: {missing_in_gt}, Missing in result: {missing_in_res}")

    if args.dry_run:
        print("[DRY-RUN] Not writing output files.")
        return

    subset_gt_path = os.path.join(args.out_dir, "subset_gt.json")
    subset_result_path = os.path.join(args.out_dir, "subset_result.json")
    save_json(subset_gt, subset_gt_path)
    save_json(pruned_res, subset_result_path)

    print("\n[NEXT STEP] After patching eval.py + task (see provided diff), run:")
    print(f"python eval.py --cfg-path <your_cfg.yaml> --res-path {subset_result_path} --gt-path {subset_gt_path} --metric")

if __name__ == "__main__":
    main()