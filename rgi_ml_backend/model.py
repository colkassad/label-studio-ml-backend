import os
import sys
from typing import List, Dict, Optional
from uuid import uuid4
import logging

import numpy as np
import torch
from PIL import Image
import cv2  # required for contour extraction

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

# ---- SAM2 imports on your PYTHONPATH
ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

LOG_SELECTED_LABEL = os.getenv("LOG_SELECTED_LABEL", "0") not in ("0", "false", "False")

# ----------------------
# Runtime configuration
# ----------------------
DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'configs/sam2.1/sam2.1_hiera_l.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')

# Polygon extraction knobs
POLY_MIN_AREA_PX = int(os.getenv('POLY_MIN_AREA_PX', '25'))             # drop tiny specks
POLY_APPROX_EPS_FRACTION = float(os.getenv('POLY_APPROX_EPS_FRACTION', '0.003'))  # simplification
POLY_RETR_MODE = os.getenv('POLY_RETR_MODE', 'EXTERNAL').upper()        # EXTERNAL or TREE

# If we can't infer a class from UI, use this default
DEFAULT_POLY_LABEL = os.getenv('DEFAULT_POLY_LABEL', 'car')

# KeyPoint negative aliases (comma-sep ENV to customize)
NEG_KEYWORDS = set([s.strip().lower() for s in os.getenv("NEG_KEYWORDS", "negative,neg,background").split(",") if s.strip()])

if DEVICE == 'cuda':
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ----------------------
# Build model
# ----------------------
sam2_checkpoint = os.path.join(ROOT_DIR, "checkpoints", MODEL_CHECKPOINT)
sam2_model = build_sam2(MODEL_CONFIG, sam2_checkpoint, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)


def _find_contours(mask_bin: np.ndarray):
    """Handle OpenCV 3/4 return signatures."""
    res = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 3:
        _, contours, hierarchy = res
    else:
        contours, hierarchy = res
    return contours, hierarchy


class NewModel(LabelStudioMLBase):
    """SAM2 → polygons for Label Studio (interactive auto-annotation)."""

    # ---------- utility & wiring ----------
    def _get_polygon_control(self):
        """Get the PolygonLabels control tuple (from_name, to_name, value)."""
        # e.g. <PolygonLabels name="labels" toName="image" ...>
        return self.get_first_tag_occurence('PolygonLabels', 'Image')

    def _get_prompt_tools(self):
        """
        from_names for KeyPointLabels and RectangleLabels if present.
        Both optional; model works with either/both.
        """
        kp = None
        box = None
        try:
            kp = self.get_first_tag_occurence('KeyPointLabels', 'Image')[0]
        except Exception:
            pass
        try:
            box = self.get_first_tag_occurence('RectangleLabels', 'Image')[0]
        except Exception:
            pass
        return kp, box

    def _ensure_size(self, mask: np.ndarray, width: int, height: int) -> np.ndarray:
        h, w = mask.shape[:2]
        if (w, h) != (width, height):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        return mask

    def _selected_polygon_label(self, context: Dict, poly_from_name: str, fallback: Optional[str]) -> Optional[str]:
        """
        Try several LS context shapes to extract the active class for PolygonLabels.
        Return None if not found so caller can apply different fallback order.
        """
        def pick_first_list_val(obj, keys):
            for k in keys:
                v = obj.get(k)
                if isinstance(v, list) and v:
                    return str(v[0])
            return None

        # flat list: ["car"]
        for k in ("selectedLabels", "selected_labels"):
            v = context.get(k)
            if isinstance(v, list) and v and isinstance(v[0], str):
                if LOG_SELECTED_LABEL:
                    logging.getLogger(__name__).debug(f"[SAM2] selected via {k} (flat list): {v[0]}")
                return str(v[0])

        # dict keyed by from_name: {"labels": ["car"]}
        for k in ("selectedLabels", "selected_labels"):
            v = context.get(k)
            if isinstance(v, dict):
                if poly_from_name in v and isinstance(v[poly_from_name], list) and v[poly_from_name]:
                    if LOG_SELECTED_LABEL:
                        logging.getLogger(__name__).debug(f"[SAM2] selected via {k}[{poly_from_name}]: {v[poly_from_name][0]}")
                    return str(v[poly_from_name][0])
                cand = pick_first_list_val(v, ("labels", "polygonlabels", "choices", "value"))
                if cand:
                    if LOG_SELECTED_LABEL:
                        logging.getLogger(__name__).debug(f"[SAM2] selected via {k} generic key: {cand}")
                    return cand

        # list of dicts: [{"from_name":"labels","labels":["car"]}]
        for k in ("selectedLabels", "selected_labels"):
            v = context.get(k)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        if item.get("from_name") in (poly_from_name, "labels"):
                            cand = pick_first_list_val(item, ("labels", "polygonlabels", "choices", "value"))
                            if cand:
                                if LOG_SELECTED_LABEL:
                                    logging.getLogger(__name__).debug(f"[SAM2] selected via {k} list-of-dicts: {cand}")
                                return cand

        # echoed in context.result (rare)
        for r in context.get("result", []):
            if r.get("from_name") == poly_from_name and isinstance(r.get("value"), dict):
                cand = pick_first_list_val(r["value"], ("polygonlabels", "labels", "choices"))
                if cand:
                    if LOG_SELECTED_LABEL:
                        logging.getLogger(__name__).debug(f"[SAM2] selected via context.result: {cand}")
                    return cand

        if LOG_SELECTED_LABEL and fallback:
            logging.getLogger(__name__).debug(f"[SAM2] no active class chip; fallback would be '{fallback}'")
        return None

    # ---------- geometry ----------
    def masks_to_polygons(
        self,
        masks: List[np.ndarray],
        probs: List[float],
        width: int,
        height: int,
        from_name: str,
        to_name: str,
        label: str
    ) -> List[Dict]:
        results = []
        retr_mode = cv2.RETR_EXTERNAL if POLY_RETR_MODE == 'EXTERNAL' else cv2.RETR_TREE

        for mask, prob in zip(masks, probs):
            # normalize mask to {0,1} uint8
            if mask.dtype != np.uint8:
                mask_bin = (mask > 0.5).astype(np.uint8)
            else:
                mask_bin = (mask > 0).astype(np.uint8)

            mask_bin = self._ensure_size(mask_bin, width, height)

            if POLY_MIN_AREA_PX > 1:
                kernel = np.ones((3, 3), np.uint8)
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)

            res = cv2.findContours(mask_bin, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
            if len(res) == 3:
                _, contours, _ = res
            else:
                contours, _ = res

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < POLY_MIN_AREA_PX:
                    continue

                peri = float(cv2.arcLength(cnt, True))
                eps = POLY_APPROX_EPS_FRACTION * peri
                approx = cv2.approxPolyDP(cnt, eps, True)
                if approx is None or len(approx) < 3:
                    continue

                poly = approx.reshape(-1, 2).astype(np.float32)
                poly[:, 0] = poly[:, 0] / width * 100.0
                poly[:, 1] = poly[:, 1] / height * 100.0

                # close polygon if needed
                if not np.allclose(poly[0], poly[-1], atol=1e-6):
                    poly = np.vstack([poly, poly[0]])

                label_id = str(uuid4())[:7]
                results.append({
                    "id": label_id,
                    "from_name": from_name,
                    "to_name": to_name,
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "points": poly.tolist(),
                        "polygonlabels": [label],
                    },
                    "score": float(prob),
                    "type": "polygonlabels",
                    "readonly": False,
                    "origin": "manual",
                })

        return results

    # ---------- SAM2 inference ----------
    def set_image(self, image_url, task_id):
        image_path = get_local_path(image_url, task_id=task_id)
        image = Image.open(image_path).convert("RGB")
        predictor.set_image(np.array(image))

    def _sam_predict(self, img_url, point_coords=None, point_labels=None, input_box=None, task=None):
        self.set_image(img_url, task.get('id'))
        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box is not None else None

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )
        order = np.argsort(scores)[::-1]
        masks = masks[order]
        scores = scores[order]
        mask = masks[0].astype(np.uint8)
        prob = float(scores[0])
        return {'masks': [mask], 'probs': [prob]}

    # ---------- main entry ----------
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        # require interaction
        if not context or not context.get('result'):
            return ModelResponse(predictions=[])

        # controls
        try:
            poly_from, to_name, value = self._get_polygon_control()
        except Exception:
            return ModelResponse(predictions=[])

        kp_from, box_from = self._get_prompt_tools()

        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        # collect prompts
        point_coords, point_labels, input_box = [], [], None
        selected_label_from_kp: Optional[str] = None

        for r in context['result']:
            t = r['type']
            fx = r['value']['x'] * image_width / 100.0
            fy = r['value']['y'] * image_height / 100.0

            if t == 'keypointlabels':  # accept from any KeyPointLabels tool
                names = r['value'].get('keypointlabels') or []
                kp_name = (names[0] if names else "").strip()
                is_neg = kp_name.lower() in NEG_KEYWORDS

                point_labels.append(0 if is_neg else 1)
                point_coords.append([int(round(fx)), int(round(fy))])

                # First positive class seen on the KP tool becomes the polygon class (unless chip present)
                if (not is_neg) and (not selected_label_from_kp) and kp_name:
                    selected_label_from_kp = kp_name

            elif t == 'rectanglelabels':
                w = r['value']['width'] * image_width / 100.0
                h = r['value']['height'] * image_height / 100.0
                input_box = [int(round(fx)), int(round(fy)), int(round(fx + w)), int(round(fy + h))]

        # class priority: positive KP label -> polygon chip -> default
        chip_label = self._selected_polygon_label(context, poly_from, fallback=None)
        chosen_label = selected_label_from_kp or chip_label or DEFAULT_POLY_LABEL

        if LOG_SELECTED_LABEL:
            logging.getLogger(__name__).debug(
                f"[SAM2] chosen_label='{chosen_label}' "
                f"(kp='{selected_label_from_kp}', chip='{chip_label}'), "
                f"points={len(point_coords)}, box={'yes' if input_box is not None else 'no'}"
            )

        # Run SAM2
        img_url = tasks[0]['data'][value]
        predictor_results = self._sam_predict(
            img_url=img_url,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0]
        )

        # Convert to polygons
        poly_results = self.masks_to_polygons(
            masks=predictor_results['masks'],
            probs=predictor_results['probs'],
            width=image_width,
            height=image_height,
            from_name=poly_from,
            to_name=to_name,
            label=chosen_label
        )

        avg_score = float(np.mean(predictor_results['probs'])) if predictor_results['probs'] else 0.0
        return ModelResponse(predictions=[{
            'result': poly_results,
            'model_version': self.get('model_version'),
            'score': avg_score
        }])
