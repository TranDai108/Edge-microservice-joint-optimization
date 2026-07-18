"""Canonical detection variant catalog.

Foundation-preserving constants only: this module centralizes naming and
variant metadata so controller/solver/build scripts stay in sync.
"""

DEFAULT_DETECTION_VARIANT = "yolo26-nano"

DETECTION_VARIANTS = [
    "yolo26-nano",
    "yolo26-small",
    "yolo26-medium",
]

DETECTION_MODEL_FILES = {
    "yolo26-nano": "yolo26n.pt",
    "yolo26-small": "yolo26s.pt",
    "yolo26-medium": "yolo26m.pt",
}

DETECTION_OFFLINE_ACCURACY = {
    "yolo26-nano": 0.48,
    "yolo26-small": 0.68,
    "yolo26-medium": 0.85,
}

DETECTION_CPU_SCALE = {
    "yolo26-nano": 0.5,
    "yolo26-small": 1.0,
    "yolo26-medium": 2.0,
}

DETECTION_MEM_SCALE = {
    "yolo26-nano": 0.7,
    "yolo26-small": 1.0,
    "yolo26-medium": 1.5,
}

DETECTION_ROLLOUT_TIMEOUT_S = {
    "yolo26-nano": 120,
    "yolo26-small": 120,
    "yolo26-medium": 180,
}

DEFAULT_GEN_AI_VARIANT = "qwen-1.5b-nano"

GEN_AI_VARIANTS = [
    "qwen-1.5b-nano",
    "llama-3b-small",
    "gemma2-2b-medium",
]

GEN_AI_OFFLINE_QUALITY = {
    "qwen-1.5b-nano": 0.40,
    "llama-3b-small": 0.75,
    "gemma2-2b-medium": 0.65,
}

GEN_AI_CPU_SCALE = {
    "qwen-1.5b-nano": 1.0,      # 1.5B / 1.5 = 1.0 (reference)
    "llama-3b-small": 2.0,       # 3.2B / 1.5 ≈ 2.0
    "gemma2-2b-medium": 1.33,    # 2.0B / 1.5 ≈ 1.33
}

GEN_AI_MEM_SCALE = {
    "qwen-1.5b-nano": 1.0,       # baseline
    "llama-3b-small": 2.0,        # 3.2B params ≈ 2x memory
    "gemma2-2b-medium": 1.4,      # 2.0B params ≈ 1.4x memory
}