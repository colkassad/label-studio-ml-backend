import os
import sys
import pathlib
from typing import List, Dict, Optional
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

import cv2  # NEW: needed for contour -> polygon extraction

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

ROOT_DIR = os.getcwd()
sys.path.insert(0, ROOT_DIR)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ----------------------
# Runtime configuration
# ----------------------
DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'configs/sam2.1/sam2.1_hiera_l.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')

# Polygon extraction tuning (override via env if needed)
POLY_MIN_AREA_PX = int(os.getenv('POLY_MIN_AREA_PX', '25'))       # filter tiny specks
POLY_APPROX_EPS_FRACTION = float(os.getenv('POLY_APPROX_EPS_FRACTION', '0.003'))  # fraction of perimeter for simplification
POLY_RETR_MODE = os.getenv('POLY_RETR_MODE', 'EXTERNAL').upper()  # EXTERNAL or TREE
# If you need holes and relations between parts, set TREE and post-process hierarchy.

# When true, if no PolygonLabels control is found in the project config, we fallback to BrushLabels
ALLOW_BRUSH_FALLBACK = os.getenv('ALLOW_BRUSH_FALLBACK', '1') not in ('0', 'false', 'False')

if DEVICE == 'cuda':
    # Use bfloat16 for speed/VRAM (Ampere+)
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ----------------------
# Build model
# ----------------------
sam2_checkpoint = str(os.path.join(ROOT_DIR, "checkpoints", MODEL_CHECKPOINT))
sam2_model = build_sam2(MODEL_CONFIG, sam2_checkpoint, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model that returns polygons for Label Studio."""

    # ----------------------
    # Helpers
    # ----------------------
    def _get_polygon_or_brush_control(self):
        """
        Prefer PolygonLabels. If not present and fallback enabled, use BrushLabels.
        Returns (from_name, to_name, value, control_type)
        """
        try:
            return (*self.get_first_tag_occurence('PolygonLabels', 'Image'), 'PolygonLabels')
        except Exception:
            if ALLOW_BRUSH_FALLBACK:
                return (*self.get_first_tag_occurence('BrushLabels', 'Image'), 'BrushLabels')
            # If no polygon and fallback disabled, re-raise
            raise

    @staticmethod
    def _ensure_h_w(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize mask to (height, width) if shapes differ."""
        h, w = mask.shape[:2]
        if (w, h) != (width, height):
            # Use nearest for labels
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        return mask

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
        """
        Convert list of binary masks into Label Studio polygon results.
        """
        results = []
        retr = cv2.RETR_EXTERNAL if POLY_RETR_MODE == 'EXTERNAL' else cv2.RETR_TREE

        for mask, prob in zip(masks, probs):
            # Normalize mask to {0,1} uint8
            if mask.dtype != np.uint8:
                # masks from SAM/SAM2 can be bool or float in {0,1}
                mask_bin = (mask > 0.5).astype(np.uint8)
            else:
                mask_bin = (mask > 0).astype(np.uint8)

            # Ensure expected size
            mask_bin = self._ensure_h_w(mask_bin, width, height)

            # Optional: slight morphology to clean tiny noise before contouring
            # (Small open operation helps remove salt noise without eroding edges much)
            if POLY_MIN_AREA_PX > 1:
                kernel = np.ones((3, 3), np.uint8)
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)

            # Find contours
            contours, hierarchy = cv2.findContours(
                mask_bin, retr, cv2.CHAIN_APPROX_SIMPLE
            )

            label_ids = []
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < POLY_MIN_AREA_PX:
                    continue

                # Simplify contour (Douglas-Peucker)
                peri = float(cv2.arcLength(cnt, True))
                eps = POLY_APPROX_EPS_FRACTION * peri
                approx = cv2.approxPolyDP(cnt, eps, True)  # already closed
                if approx is None or len(approx) < 3:
                    continue

                # Squeeze to Nx2
                polygon = approx.reshape(-1, 2)

                # Convert to percent coords
                # Note: Label Studio expects [x%, y%], 0..100
                polygon_pct = np.empty_like(polygon, dtype=np.float32)
                polygon_pct[:, 0] = polygon[:, 0] / width * 100.0
                polygon_pct[:, 1] = polygon[:, 1] / height * 100.0

                # Ensure closed: LS usually accepts open list; we close to be safe
                # by appending first point if last != first (within a small tolerance)
                if not np.allclose(polygon_pct[0], polygon_pct[-1], atol=1e-6):
                    polygon_pct = np.vstack([polygon_pct, polygon_pct[0]])

                label_id = str(uuid4())[:7]
                results.append({
                    "id": label_id,
                    "from_name": from_name,
                    "to_name": to_name,
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "points": polygon_pct.tolist(),
                        "polygonlabels": [label],
                    },
                    "score": float(prob),
                    "type": "polygonlabels",
                    "readonly": False,
                    "origin": "manual",
                })
                label_ids.append(label_id)

            # If you want to express relations between polygons (e.g., parts),
            # you can add relation objects here similar to your original code.
            # For now we skip relations to keep LS UI editing simple.

        return results

    def get_results_brush(
        self, masks, probs, width, height, from_name, to_name, label
    ):
        """Original brush (RLE) output kept as a fallback."""
        results = []
        total_prob = 0.0
        for mask, prob in zip(masks, probs):
            label_id = str(uuid4())[:4]
            mask_u8 = (mask > 0.5).astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
            mask_u8 = self._ensure_h_w(mask_u8, width, height)

            rle = brush.mask2rle(mask_u8)
            total_prob += float(prob)
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [label],
                },
                'score': float(prob),
                'type': 'brushlabels',
                'readonly': False
            })

        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]

    def set_image(self, image_url, task_id):
        image_path = get_local_path(image_url, task_id=task_id)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        predictor.set_image(image)

    def _sam_predict(self, img_url, point_coords=None, point_labels=None, input_box=None, task=None):
        self.set_image(img_url, task.get('id'))
        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            multimask_output=True
        )
        # sort by score desc and pick top-1 (you can return multiple if desired)
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        mask = masks[0].astype(np.uint8)  # bool/float -> uint8
        prob = float(scores[0])
        return {
            'masks': [mask],
            'probs': [prob]
        }

    # ----------------------
    # Main prediction entry
    # ----------------------
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        Returns the predicted polygon(s) for an interactive point/box placed in Label Studio.
        """
        try:
            from_name, to_name, value, control_type = self._get_polygon_or_brush_control()
        except Exception:
            # If nothing found at all, bail gracefully
            return ModelResponse(predictions=[])

        if not context or not context.get('result'):
            # no interaction yet
            return ModelResponse(predictions=[])

        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        # Collect interaction context
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None

        for ctx in context['result']:
            x = ctx['value']['x'] * image_width / 100.0
            y = ctx['value']['y'] * image_height / 100.0
            ctx_type = ctx['type']

            # Label chosen by the user (from the tool they used)
            # Works for keypointlabels/rectanglelabels, etc.
            if ctx['value'].get(ctx_type):
                selected_label = ctx['value'][ctx_type][0]

            if ctx_type == 'keypointlabels':
                point_labels.append(int(ctx.get('is_positive', 0)))
                point_coords.append([int(round(x)), int(round(y))])
            elif ctx_type == 'rectanglelabels':
                box_width = ctx['value']['width'] * image_width / 100.0
                box_height = ctx['value']['height'] * image_height / 100.0
                input_box = [
                    int(round(x)),
                    int(round(y)),
                    int(round(x + box_width)),
                    int(round(y + box_height))
                ]

        img_url = tasks[0]['data'][value]
        predictor_results = self._sam_predict(
            img_url=img_url,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0]
        )

        masks = predictor_results['masks']
        probs = predictor_results['probs']

        # If we have a polygon control, return polygons; otherwise brush fallback
        if control_type == 'PolygonLabels':
            poly_results = self.masks_to_polygons(
                masks=masks,
                probs=probs,
                width=image_width,
                height=image_height,
                from_name=from_name,
                to_name=to_name,
                label=selected_label or 'Object'
            )
            avg_score = float(np.mean(probs)) if len(probs) else 0.0
            predictions = [{
                'result': poly_results,
                'model_version': self.get('model_version'),
                'score': avg_score
            }]
        else:
            predictions = self.get_results_brush(
                masks=masks,
                probs=probs,
                width=image_width,
                height=image_height,
                from_name=from_name,
                to_name=to_name,
                label=selected_label or 'Object'
            )

        return ModelResponse(predictions=predictions)
