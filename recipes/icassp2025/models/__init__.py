from torch import nn
from typing import List, Type

from . import model_2024


_NAME_TO_MODEL_CLASS = {
    'EstimateWienerFilter': model_2024.EstimateWienerFilter,
    'CorrectPhase': model_2024.CorrectPhase,
    'FrameWiseCorrectPhaseWithIFDMask': model_2024.FrameWiseCorrectPhaseWithIFDMask,
}


def get_model(model_name: str) -> Type[nn.Module]:
    return _NAME_TO_MODEL_CLASS[model_name]


def get_model_names() -> List[str]:
    return _NAME_TO_MODEL_CLASS.keys()


# Contain models with `enhance` method
_NAME_TO_ENHANCEMENT_MODEL_CLASS = {
    'EstimateWienerFilter': model_2024.EstimateWienerFilter,
    'CorrectPhase': model_2024.CorrectPhase,
    'FrameWiseCorrectPhaseWithIFDMask': model_2024.FrameWiseCorrectPhaseWithIFDMask,
}


def get_enhancement_model(model_name: str) -> Type[nn.Module]:
    return _NAME_TO_ENHANCEMENT_MODEL_CLASS[model_name]


def get_enhancement_model_names() -> List[str]:
    return _NAME_TO_ENHANCEMENT_MODEL_CLASS.keys()
