"""
Global configuration settings for AI Virtual Try-On
"""

import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Assets paths
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
GARMENTS_DIR = os.path.join(ASSETS_DIR, "garments")

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Pose detection settings
POSE_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

# Garment scaling settings
SHOULDER_SCALE_FACTOR = 1.4
TORSO_SCALE_FACTOR = 1.2

# Physics settings
GRAVITY = 0.4
DAMPING = 0.9
SMOOTHING_FACTOR = 0.7

# Rendering settings
OVERLAY_ALPHA = 0.9

# Supported garment formats
SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg"]