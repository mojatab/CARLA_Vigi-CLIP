#!/usr/bin/env python3

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Driver Monitoring System with Intel RealSense Integration.

This script extends the CARLA manual-control example by adding an Intel
RealSense camera stream for in-vehicle driver monitoring. It supports multiple
DMS modules, including gaze estimation, face anti-spoofing, and driver-action
recognition, and can record both the CARLA camera and RealSense camera streams.

Controls
--------
H or ?      Show/hide help
M           Cycle DMS mode: Gaze -> Anti-Spoofing -> Action -> ALL
T           Start/stop synchronized CARLA + RealSense video recording
R           Toggle CARLA frame recording
P           Toggle CARLA autopilot
TAB         Switch camera position
ESC / Ctrl+Q Exit

Notes
-----
- Configure your steering wheel in ``wheel_config.ini``.
- Start the CARLA server before running this script.
- Model paths can be configured through command-line arguments or environment
  variables.
"""

from __future__ import print_function

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import enum
import importlib.util as _ilu
import logging
import math
import random
import re
import weakref
import cv2
import threading

# RealSense imports
import pyrealsense2 as rs
import numpy as np

# Gaze estimation imports
import torch
import torch.nn.functional as F
import mediapipe as mp
import networkx as nx
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention, TransformerConv

os.makedirs("output", exist_ok=True)

# Third-party module paths (anti-spoof model.bulid_model + ViFi-CLIP utils)
_SPOOF_ROOT = os.environ.get('SPOOF_ROOT', '/home/michael/LFAS-NewBackbone Ablation Study')
_VIFI_ROOT  = os.environ.get('VIFI_ROOT', '/home/michael/CARLA_UE5/ViFi-CLIP')
for _p in (_SPOOF_ROOT, _VIFI_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ==============================================================================
# -- DMS Mode ------------------------------------------------------------------
# ==============================================================================

class DmsMode(enum.Enum):
    GAZE         = "Gaze"
    ANTISPOOFING = "Anti-Spoofing"
    ACTION       = "Action"
    ALL          = "ALL"

    def next(self):
        members = list(DmsMode)
        return members[(members.index(self) + 1) % len(members)]

# ==============================================================================
# -- TGGNet Gaze Model ---------------------------------------------------------
# ==============================================================================

class TransformerNet(torch.nn.Module):
    def __init__(self, num_node_features: int):
        super().__init__()
        head_dim1, head_dim2, head_dim3, head_dim4 = 64, 32, 16, 8
        self.conv1 = TransformerConv(num_node_features, head_dim1 * 8)
        self.conv2 = TransformerConv(head_dim1 * 8, head_dim2 * 8)
        self.conv3 = TransformerConv(head_dim2 * 8, head_dim3 * 4)
        self.conv4 = TransformerConv(head_dim3 * 4, head_dim4 * 4)
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(head_dim4 * 4, 1))
        self.fc = torch.nn.Linear(head_dim4 * 4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = self.att_pool(x, data.batch)
        return self.fc(x)


GAZE_EDGES = [(468, node) for node in range(478) if node != 468]
GAZE_EDGES += [(473, node) for node in range(478) if node != 473]
GAZE_EDGES.extend([
    (471, 159), (159, 469), (469, 145), (145, 471),
    (476, 475), (475, 474), (474, 477), (477, 476),
    (1, 33),  (1, 173), (1, 162), (1, 263), (1, 398), (1, 368),
    (33, 246),
    (146, 161), (161, 160), (160, 150), (150, 158), (158, 157), (157, 173),
    (173, 155), (155, 154), (154, 153), (153, 145), (145, 144), (144, 163),
    (163, 7),  (7, 33),
    (398, 384), (384, 385), (385, 386), (386, 387), (387, 388), (388, 263),
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 398),
])

ARROW_SCALE = 120


def _load_gaze_model(model_path: str, device: torch.device) -> TransformerNet:
    model = TransformerNet(num_node_features=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[GazeProcessor] Model loaded from: {model_path}  (device: {device})")
    return model


def _make_face_landmarker(task_model_path: str):
    base_opts = mp_python.BaseOptions(model_asset_path=task_model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def _extract_landmarks(image_bgr: np.ndarray, landmarker):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, None
    raw = [(lm.x, lm.y, lm.z) for lm in result.face_landmarks[0]]
    face_xs = [lm[0] for lm in raw[:468]]
    face_ys = [lm[1] for lm in raw[:468]]
    x_min, x_max = min(face_xs), max(face_xs)
    y_min, y_max = min(face_ys), max(face_ys)
    px = (x_max - x_min) * 0.10
    py = (y_max - y_min) * 0.10
    x_min -= px;  x_max += px
    y_min -= py;  y_max += py
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    z_scale = x_range
    normalized = [((lm[0] - x_min) / x_range,
                   (lm[1] - y_min) / y_range,
                   lm[2] / z_scale) for lm in raw]
    return raw, normalized


def _build_graph(landmarks: list, edges: list, device: torch.device) -> Data:
    G = nx.Graph()
    for i, pos in enumerate(landmarks):
        G.add_node(i, pos=pos)
    G.add_edges_from(edges)
    x = torch.tensor([G.nodes[i]["pos"] for i in G.nodes()], dtype=torch.float)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data.to(device)


def _draw_gaze_hud(frame: np.ndarray, gaze_x: float, gaze_y: float) -> None:
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    label = f"gaze=({gaze_x:+.3f}, {gaze_y:+.3f})"
    cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


def _draw_no_face(frame: np.ndarray) -> None:
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(frame, "No face detected", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)


def _draw_gaze_arrows(frame: np.ndarray, landmarks: list,
                      gaze_x: float, gaze_y: float, w: int, h: int) -> None:
    dx = int(gaze_x * ARROW_SCALE)
    dy = int(-gaze_y * ARROW_SCALE)
    for lm_idx in (468, 473):
        sx = int(landmarks[lm_idx][0] * w)
        sy = int(landmarks[lm_idx][1] * h)
        cv2.circle(frame, (sx, sy), 6, (0, 255, 0), -1)
        cv2.arrowedLine(frame, (sx, sy), (sx + dx, sy + dy),
                        (0, 0, 255), 3, tipLength=0.2)


def _draw_gaze_compass(frame: np.ndarray, gaze_x: float, gaze_y: float) -> None:
    h, w = frame.shape[:2]
    r      = 60
    cx, cy = w - r - 15, h - r - 15
    cv2.circle(frame, (cx, cy), r, (40, 40, 40), -1)
    cv2.circle(frame, (cx, cy), r, (160, 160, 160), 2)
    for angle in (0, 90, 180, 270):
        rad = np.deg2rad(angle)
        tx  = int(cx + (r - 8) * np.sin(rad))
        ty  = int(cy - (r - 8) * np.cos(rad))
        cv2.circle(frame, (tx, ty), 3, (100, 100, 100), -1)
    needle_len = r - 10
    nx_ = int(cx + gaze_x * needle_len)
    ny_ = int(cy + (-gaze_y) * needle_len)
    cv2.arrowedLine(frame, (cx, cy), (nx_, ny_), (0, 0, 255), 3, tipLength=0.35)
    cv2.putText(frame, "GAZE", (cx - 20, cy + r + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# Load realtime_test module from its original location so __file__-based
# project_root resolution (used by load_model) finds the model/ package correctly
_RT_PATH = os.environ.get(
    'REALTIME_TEST_PATH',
    os.path.join(_SPOOF_ROOT, 'realtime testing', 'realtime_test.py')
)
_rt_spec = _ilu.spec_from_file_location("realtime_test", _RT_PATH)
_rt = _ilu.module_from_spec(_rt_spec)
try:
    _rt_spec.loader.exec_module(_rt)
    _RT_AVAILABLE = True
except Exception as _rt_err:
    _RT_AVAILABLE = False
    print(f"[SpoofingProcessor] realtime_test load failed: {_rt_err}")


# ==============================================================================
# -- GazeProcessor -------------------------------------------------------------
# ==============================================================================

class GazeProcessor(object):
    def __init__(self, model_path: str, task_model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_gaze_model(model_path, self.device)
        self.landmarker = _make_face_landmarker(task_model_path)
        print(f"[GazeProcessor] Ready on device: {self.device}")

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run gaze inference and draw annotations. Returns annotated BGR frame."""
        raw_lm, norm_lm = _extract_landmarks(frame_bgr, self.landmarker)
        h, w = frame_bgr.shape[:2]
        display = frame_bgr.copy()
        gaze_vector = None
        if norm_lm is not None:
            data = _build_graph(norm_lm, GAZE_EDGES, self.device)
            with torch.no_grad():
                out = self.model(data)
            gaze_vector = out.cpu().numpy()[0]
        if gaze_vector is not None and raw_lm is not None:
            _draw_gaze_arrows(display, raw_lm, gaze_vector[0], gaze_vector[1], w, h)
            _draw_gaze_compass(display, gaze_vector[0], gaze_vector[1])
            _draw_gaze_hud(display, gaze_vector[0], gaze_vector[1])
        else:
            _draw_no_face(display)
        return display

    def cleanup(self):
        self.landmarker.close()


# ==============================================================================
# -- SpoofingProcessor ---------------------------------------------------------
# ==============================================================================

class SpoofingProcessor(object):
    def __init__(self, checkpoint: str):
        if not _RT_AVAILABLE:
            raise RuntimeError("realtime_test module could not be loaded")
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)

        # Build minimal args namespace matching realtime_test.parse_args() defaults
        class _Args:
            pass
        args = _Args()
        args.checkpoint        = checkpoint
        args.model             = 'ShffleNetV2_hd_v1_hybrid_d'  # checkpoint name doesn't encode model type
        args.is_multi          = True
        args.image_modality    = 'color'
        args.guidance_modality = 'depth'
        args.adaptive_guidance = False       # auto-detected inside load_model
        args.device            = device_str
        args.depth_min         = 300
        args.depth_max         = 800
        args.depth_validity_override = 0.4
        args.smooth_window     = 5
        args.face_scale        = 1.3
        args.haar_scale        = 1.1
        args.haar_min_neighbors = 5
        args.haar_min_size     = 60

        self.model        = _rt.load_model(args, self.device)
        self.face_detector = _rt.FaceDetector(
            scale_factor=args.haar_scale,
            min_neighbors=args.haar_min_neighbors,
            min_size=args.haar_min_size)
        self.preprocessor = _rt.FramePreprocessor(args)
        self.smoother     = _rt.PredictionSmoother(args.smooth_window)
        self.renderer     = _rt.OverlayRenderer()
        self._face_scale  = args.face_scale
        print(f"[SpoofingProcessor] Ready on device: {self.device}")

    def process(self, color_bgr: np.ndarray,
                depth_u16: np.ndarray,
                ir_u8: np.ndarray) -> np.ndarray:
        """Run face anti-spoofing inference and draw annotations."""
        gray  = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detect(gray)
        display = color_bgr.copy()
        face_bbox = None
        gw = None

        if faces:
            x, y, w, h = faces[0]
            face_bbox = (x, y, w, h)
            tensor = self.preprocessor.preprocess(
                color_bgr, depth_u16, ir_u8, x, y, w, h, scale=self._face_scale)
            if tensor is not None:
                spoof_prob, gw = _rt.run_inference(self.model, tensor, self.device)
                self.smoother.update(spoof_prob)

        # Save top-left patch so we can erase the FPS text that renderer draws
        fps_patch = display[0:45, 0:160].copy()
        self.renderer.draw(
            display,
            face_bbox,
            self.smoother.get(),
            fps=0,
            guidance_weights=gw)
        # Restore region to remove "FPS: 0.0" text
        display[0:45, 0:160] = fps_patch
        return display

    def cleanup(self):
        pass  # no resources to release


# ==============================================================================
# -- ActionProcessor -----------------------------------------------------------
# ==============================================================================

_ACTION_MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
_ACTION_STD  = np.array([58.395,  57.12,   57.375],  dtype=np.float32)


_DRIVER_ACTION_CLASSES = [
    "Driving is normal and the driver is Safely driving",
    "Driving is abnormal because the driver is Doing hair and makeup",
    "Driving is abnormal because the driver is Adjusting radio",
    "Driving is abnormal because the driver is GPS operating",
    "Driving is abnormal because the driver is Writing message using right hand",
    "Driving is abnormal because the driver is Writing message using left hand",
    "Driving is abnormal because the driver is Talking phone using right hand",
    "Driving is abnormal because the driver is Talking phone using left hand",
    "Driving is abnormal because the driver is Having picture",
    "Driving is abnormal because the driver is Talking to passenger",
    "Driving is abnormal because the driver is Singing or dancing",
    "Driving is abnormal because the driver is Fatigue and somnolence",
    "Driving is abnormal because the driver is Drinking using right hand",
    "Driving is abnormal because the driver is Drinking using left hand",
    "Driving is abnormal because the driver is Reaching behind",
    "Driving is abnormal because the driver is Smoking",
]

# Highlight these as dangerous actions
_DANGER_ACTIONS = {
    "Writing message using right hand", "Writing message using left hand",
    "Talking phone using right hand",   "Talking phone using left hand",
    "Drinking using right hand",        "Drinking using left hand",
    "Smoking",
}


def _short_action_label(full_name: str) -> str:
    """Strip the long prefix for on-screen display."""
    return (full_name
            .replace("Driving is normal and the driver is ", "")
            .replace("Driving is abnormal because the driver is ", ""))


class ActionProcessor(object):
    def __init__(self, vifi_root: str, checkpoint: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = _DRIVER_ACTION_CLASSES

        # Lazy import ViFi-CLIP (vifi_root already in sys.path)
        from utils.config import get_config
        from trainers.vificlip import returnCLIP

        class _Args:
            config              = os.path.join(vifi_root, 'configs/zero_shot/train/k400/16_16_vifi_clip.yaml')
            output              = os.path.join(vifi_root, 'output_vificlip')
            resume              = checkpoint
            only_test           = True
            opts                = None
            batch_size          = None
            pretrained          = None
            accumulation_steps  = None
            local_rank          = 0

        config = get_config(_Args())
        _logger = logging.getLogger('ActionProcessor')
        self.model = returnCLIP(config, logger=_logger,
                                class_names=self.class_names).float().to(self.device).eval()

        # Load checkpoint — strip DataParallel prefix, drop token embeddings
        # that don't exist in this model config (complete_text_embeddings is kept
        # because it matches the 16-class shape exactly)
        ckpt = torch.load(checkpoint, map_location='cpu')
        _bad_keys = {
            'module.prompt_learner.token_prefix',
            'module.prompt_learner.token_suffix',
            'prompt_learner.token_prefix',
            'prompt_learner.token_suffix',
        }
        state_dict = {k: v for k, v in ckpt.items() if k not in _bad_keys}
        state_dict = {(k[7:] if k.startswith('module.') else k): v
                      for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.float()   # re-cast after checkpoint (checkpoint may contain fp16 weights)
        # CLIP stores dtype as an instance attribute (not a parameter), so .float()
        # doesn't update it. Patch every module that has a 'dtype' attr to float32
        # so that forward()'s image.type(self.dtype) doesn't cast inputs back to fp16.
        for m in self.model.modules():
            if hasattr(m, 'dtype'):
                m.dtype = torch.float32
        self.model.eval()

        self.frame_buffer      = collections.deque(maxlen=32)
        self._last_label       = "..."
        self._last_conf        = 0.0
        self._frames_since_inf = 0
        self.INFER_EVERY       = 16
        print(f"[ActionProcessor] Ready — {len(self.class_names)} classes, device: {self.device}")

    def process(self, color_bgr: np.ndarray) -> np.ndarray:
        """Accumulate frames, run inference every INFER_EVERY frames, draw action label."""
        # Preprocess: BGR → RGB, resize 224×224, normalize
        frame = cv2.resize(color_bgr, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = (frame - _ACTION_MEAN) / _ACTION_STD  # (224, 224, 3)
        self.frame_buffer.append(frame)
        self._frames_since_inf += 1

        if (len(self.frame_buffer) == 32 and
                self._frames_since_inf >= self.INFER_EVERY):
            # Shape: [1, 32, 3, 224, 224]
            video = np.stack(list(self.frame_buffer))            # (32, 224, 224, 3)
            video = video.transpose(0, 3, 1, 2)                  # (32, 3, 224, 224)
            tensor = torch.from_numpy(video).unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                logits = self.model(tensor)
            probs = logits.softmax(dim=-1)[0]
            idx = int(probs.argmax().item())
            self._last_label = self.class_names[idx]
            self._last_conf  = float(probs[idx].item())
            self._frames_since_inf = 0

        display = color_bgr.copy()
        w = display.shape[1]
        if len(self.frame_buffer) < 32:
            text = f"Buffering... ({len(self.frame_buffer)}/32)"
            color = (180, 180, 0)
        else:
            short = _short_action_label(self._last_label)
            is_danger = short in _DANGER_ACTIONS
            text = f"{short} {self._last_conf:.0%}"
            color = (0, 0, 255) if is_danger else (0, 255, 0)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = w - tw - 10
        ty = 54  # below gaze HUD bar (34px) with margin
        cv2.rectangle(display, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(display, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return display

    def cleanup(self):
        pass  # no resources to release


try:
    import pygame
    from pygame.locals import (
        KMOD_CTRL, KMOD_SHIFT, K_0, K_9, K_BACKQUOTE, K_BACKSPACE, K_COMMA,
        K_DOWN, K_ESCAPE, K_F1, K_LEFT, K_PERIOD, K_RIGHT, K_SLASH, K_SPACE,
        K_TAB, K_UP, K_a, K_c, K_d, K_h, K_m, K_p, K_q, K_r, K_s, K_t, K_w
    )
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy is installed')

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser


# ==============================================================================
# -- RealSense Manager ---------------------------------------------------------
# ==============================================================================

class RealSenseManager(object):
    def __init__(self, gaze_processor=None, spoof_processor=None, action_processor=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color,    640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth,    640, 480, rs.format.z16,  30)
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        self.pipeline.start(self.config)
        self._align = rs.align(rs.stream.color)

        self.recording = False
        self.video_writer = None
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()

        self.gaze_processor   = gaze_processor
        self.spoof_processor  = spoof_processor
        self.action_processor = action_processor
        self.dms_mode         = DmsMode.GAZE

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def cycle_mode(self) -> DmsMode:
        self.dms_mode = self.dms_mode.next()
        return self.dms_mode

    def _capture_loop(self):
        """Continuously capture and annotate frames from RealSense in background thread"""
        while self.running:
            try:
                frames  = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self._align.process(frames)

                color_frame = aligned.get_color_frame()
                if not color_frame:
                    continue

                depth_frame = aligned.get_depth_frame()
                ir_frame    = aligned.get_infrared_frame(1)

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
                ir_image    = np.asanyarray(ir_frame.get_data())    if ir_frame    else None

                mode    = self.dms_mode
                display = color_image.copy()

                if mode in (DmsMode.ANTISPOOFING, DmsMode.ALL) and self.spoof_processor:
                    try:
                        display = self.spoof_processor.process(display, depth_image, ir_image)
                    except Exception as e:
                        print(f"Spoof processing error: {e}")

                if mode in (DmsMode.GAZE, DmsMode.ALL) and self.gaze_processor:
                    try:
                        display = self.gaze_processor.process(display)
                    except Exception as e:
                        print(f"Gaze processing error: {e}")

                if mode in (DmsMode.ACTION, DmsMode.ALL) and self.action_processor:
                    try:
                        display = self.action_processor.process(display)
                    except Exception as e:
                        print(f"Action processing error: {e}")

                with self.lock:
                    self.latest_frame = display.copy()
                    if self.recording and self.video_writer is not None:
                        self.video_writer.write(display)

            except Exception as e:
                print(f"RealSense capture error: {e}")

    def start_recording(self):
        """Start recording RealSense video"""
        if not self.recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/realsense_rgb_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))

            self.recording = True
            print(f"🎥 Started RealSense recording: {filename}")
            return filename
        return None

    def stop_recording(self):
        """Stop recording RealSense video"""
        if self.recording:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            print("⏹️ Stopped RealSense recording.")

    def get_latest_frame(self):
        """Get the most recent frame"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.video_writer is not None:
            self.video_writer.release()
        self.pipeline.stop()


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, actor_filter, realsense_manager):
        self.world = carla_world
        self.hud = hud
        self.realsense_manager = realsense_manager
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self.realsense_manager)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.hud.notification(get_actor_display_name(self.player))

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification(f'Weather: {preset[1]}')
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor]
        for s in sensors:
            if s is not None:
                s.stop()
                s.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- DualControl ---------------------------------------------------------------
# ==============================================================================

class DualControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")

        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please connect just one joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(self._parser.get('Logitech G920 Driving Force Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(self._parser.get('Logitech G920 Driving Force Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('Logitech G920 Driving Force Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('Logitech G920 Driving Force Racing Wheel', 'reverse'))

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h:
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == K_t:
                    world.camera_manager.toggle_video_recording()
                elif event.key == K_m:
                    new_mode = world.camera_manager.realsense_manager.cycle_mode()
                    world.hud.notification(f'DMS Mode: {new_mode.value}')
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification(f'Autopilot {"On" if self._autopilot_enabled else "Off"}')

        if not self._autopilot_enabled:
            self._parse_vehicle_wheel()
            world.player.apply_control(self._control)

    def _parse_vehicle_wheel(self):
        num_axes = self._joystick.get_numaxes()
        js_inputs = [float(self._joystick.get_axis(i)) for i in range(num_axes)]
        js_buttons = [float(self._joystick.get_button(i)) for i in range(self._joystick.get_numbuttons())]

        K1 = 1.0
        steer_cmd = K1 * math.tan(1.1 * js_inputs[self._steer_idx])

        K2 = 1.6
        throttle_cmd = K2 + (2.05 * math.log10(-0.7 * js_inputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        throttle_cmd = min(max(throttle_cmd, 0), 1)

        brake_cmd = 1.6 + (2.05 * math.log10(-0.7 * js_inputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        brake_cmd = min(max(brake_cmd, 0), 1)

        self._control.steer = steer_cmd
        self._control.throttle = throttle_cmd
        self._control.brake = brake_cmd
        self._control.reverse = bool(js_buttons[self._reverse_idx])

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            # Position HUD on the right side instead of left
            info_x_offset = self.dim[0] - 260
            display.blit(info_surface, (info_x_offset, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(info_x_offset + x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((info_x_offset + bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((info_x_offset + bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((info_x_offset + bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((info_x_offset + bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (info_x_offset + 8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================

class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud, realsense_manager):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.realsense_manager = realsense_manager
        self.recording = False
        self.video_writer = None
        self.video_file = f'output/carla_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        self.fps = 18
        self.frame_size = (hud.dim[0], hud.dim[1])
        self._camera_transforms = [
            carla.Transform(carla.Location(x=0.35, y=-0.4, z=1.146)),
            carla.Transform(carla.Location(x=1.5, z=0.1))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                bp.set_attribute('fov', '150')
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification(f"Frame Recording {'On' if self.recording else 'Off'}")

    def toggle_video_recording(self):
        """Toggle recording for BOTH CARLA and RealSense cameras"""
        if self.video_writer is None:
            # Start CARLA recording
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_file = f'output/carla_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_file, fourcc, self.fps, self.frame_size)

            # Start RealSense recording
            self.realsense_manager.start_recording()

            self.hud.notification("🎥 Recording BOTH cameras (T pressed)")
        else:
            # Stop CARLA recording
            self.video_writer.release()
            self.video_writer = None

            # Stop RealSense recording
            self.realsense_manager.stop_recording()

            self.hud.notification("⏹️ Stopped recording BOTH cameras")

    def render(self, display):
        # Render main CARLA camera view
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

        # Render RealSense camera in top-left corner
        realsense_frame = self.realsense_manager.get_latest_frame()
        if realsense_frame is not None:
            # Convert BGR (OpenCV) to RGB (Pygame)
            realsense_frame_rgb = cv2.cvtColor(realsense_frame, cv2.COLOR_BGR2RGB)

            # Create pygame surface from the frame
            # Original size is 640x480, let's scale it down for overlay
            overlay_width = 640
            overlay_height = 480
            realsense_frame_resized = cv2.resize(realsense_frame_rgb, (overlay_width, overlay_height))

            # Convert to pygame surface (need to swap axes for pygame)
            realsense_surface = pygame.surfarray.make_surface(realsense_frame_resized.swapaxes(0, 1))

            # Draw border around the overlay
            border_color = (255, 255, 255)  # White border
            border_thickness = 2

            # Create a surface with border
            bordered_surface = pygame.Surface((overlay_width + border_thickness * 2,
                                              overlay_height + border_thickness * 2))
            bordered_surface.fill(border_color)
            bordered_surface.blit(realsense_surface, (border_thickness, border_thickness))

            # Blit to top-left corner (with small offset from edge)
            offset_x = 10
            offset_y = 10
            display.blit(bordered_surface, (offset_x, offset_y))

            # Add label
            font = pygame.font.Font(pygame.font.get_default_font(), 14)
            mode_name = self.realsense_manager.dms_mode.value
            label = font.render(f'DMS: {mode_name}', True, (255, 255, 255))
            label_bg = pygame.Surface((label.get_width() + 10, label.get_height() + 4))
            label_bg.fill((0, 0, 0))
            label_bg.set_alpha(180)
            display.blit(label_bg, (offset_x + border_thickness,
                                   offset_y + overlay_height + border_thickness - label.get_height() - 4))
            display.blit(label, (offset_x + border_thickness + 5,
                                offset_y + overlay_height + border_thickness - label.get_height() - 2))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        frame = None
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data).astype(np.int32)
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            frame = array

        if self.video_writer is not None and frame is not None:
            try:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame_bgr)
            except Exception as e:
                print("CARLA video write error:", e)

        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    realsense_manager = None

    try:
        # Initialize gaze processor
        gaze_processor = None
        if not args.disable_gaze:
            try:
                gaze_processor = GazeProcessor(
                    model_path=args.gaze_model,
                    task_model_path=args.face_landmarker
                )
            except Exception as e:
                print(f"Warning: Gaze processor failed to load, falling back to plain RGB: {e}")

        # Initialize anti-spoofing processor
        spoof_processor = None
        if not args.disable_spoofing:
            try:
                spoof_processor = SpoofingProcessor(checkpoint=args.spoof_checkpoint)
            except Exception as e:
                print(f"Warning: SpoofingProcessor failed to load: {e}")

        # Initialize action recognition processor
        action_processor = None
        if not args.disable_action:
            try:
                action_processor = ActionProcessor(
                    vifi_root=args.vifi_root,
                    checkpoint=args.action_checkpoint
                )
            except Exception as e:
                print(f"Warning: ActionProcessor failed to load: {e}")

        # Initialize RealSense camera
        print("Initializing RealSense camera...")
        realsense_manager = RealSenseManager(
            gaze_processor=gaze_processor,
            spoof_processor=spoof_processor,
            action_processor=action_processor)
        print("RealSense camera initialized successfully!")

        # Initialize CARLA
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter, realsense_manager)
        controller = DualControl(world, args.autopilot)
        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        # Cleanup
        if world is not None:
            try:
                if world.camera_manager.video_writer:
                    world.camera_manager.video_writer.release()
                # Disable autopilot and stop sensors cleanly
                if world.player is not None:
                    world.player.set_autopilot(False)
                if world.camera_manager.sensor is not None:
                    world.camera_manager.sensor.stop()
                world.destroy()
            except Exception as e:
                print("CARLA cleanup error:", e)

        # Cleanup RealSense
        if realsense_manager is not None:
            try:
                realsense_manager.cleanup()
                print("RealSense camera cleaned up successfully!")
            except Exception as e:
                print("RealSense cleanup error:", e)

        # Cleanup DMS processors
        for _proc, _name in [(gaze_processor, 'Gaze'),
                             (spoof_processor, 'Spoofing'),
                             (action_processor, 'Action')]:
            if _proc is not None:
                try:
                    _proc.cleanup()
                except Exception as e:
                    print(f"{_name} processor cleanup error: {e}")

        pygame.quit()
        sys.exit(0)


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='CARLA Manual Control with RealSense Recording and Display')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('-p', '--port', default=2000, type=int)
    parser.add_argument('-a', '--autopilot', action='store_true')
    parser.add_argument('--res', default='2560x1440') # 3 screens 7680x1440
    parser.add_argument('--filter', default='vehicle.taxi.ford')

    # DMS model paths. Defaults match the original development layout but can be
    # overridden for a public repository or another workstation.
    default_gaze_demo = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../GazeTGGNet-main/Demo'))
    parser.add_argument('--gaze-model', default=os.path.join(default_gaze_demo, 'trained_model_No_Or.pt'))
    parser.add_argument('--face-landmarker', default=os.path.join(default_gaze_demo, 'face_landmarker.task'))
    parser.add_argument('--spoof-checkpoint', default='/home/michael/CARLA_UE5/Face Anit-Spoofing/test_min_acer_model_20260307_13_14_45.pth')
    parser.add_argument('--vifi-root', default=_VIFI_ROOT)
    parser.add_argument('--action-checkpoint', default='/home/michael/CARLA_UE5/ViFi-CLIP/vifi_clip_finetuned_LAST02.pth')
    parser.add_argument('--disable-gaze', action='store_true', help='Disable gaze-estimation module')
    parser.add_argument('--disable-spoofing', action='store_true', help='Disable face anti-spoofing module')
    parser.add_argument('--disable-action', action='store_true', help='Disable driver-action recognition module')

    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("CARLA Driver Monitoring System with RealSense Integration")
    print("="*60)
    print("Press 'M' to cycle DMS mode:")
    print("  Gaze → Anti-Spoofing → Action → ALL")
    print("Press 'T' to start/stop recording BOTH cameras")
    print("Videos saved in 'output/' folder")
    print("="*60)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
