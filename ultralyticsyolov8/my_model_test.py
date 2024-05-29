import contextlib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from ultralytics import RTDETR, YOLO
from ultralytics.cfg import MODELS, TASKS, TASK2DATA
from ultralytics.data.build import load_inference_source
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    Retry,
    checks,
)
from ultralytics.utils.downloads import download, is_url
from ultralytics.utils.torch_utils import TORCH_1_9
from tests import CFG, IS_TMP_WRITEABLE, MODEL, SOURCE, TMP

CFG='ultralytics/cfg/models/v8/seaships_CBAM.yaml'


def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG)
    model(SOURCE)  # also test no source and augment