from enum import Enum
from typing import Optional
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    SchedulerMixin,
)

from custom_nodes.lightdream_model.scheduling_ddim import DDIMScheduler

class NoiseScheduler(Enum):
    DDIM = 'DDIM'
    DPM = 'DPM'
    DPM_PP_2M = 'DPM++2M'
    DPM_PP_2M_K = 'DPM++2M_K'
    EULER = 'EULER'
    EULER_A = 'EULER_ANCESTRAL'
    EULER_K = 'EULER_K'
    LMS = 'LMS'
    PNDM = 'PNDM'
    UNIPC_M = 'UNIPC_M'

    @classmethod
    def _missing_(cls, value: str) -> Optional['NoiseScheduler']:
        for member in cls:
            if member.name.lower() == value.lower():
                return member
        return None

def create_scheduler(scheduler: NoiseScheduler) -> SchedulerMixin:
    if scheduler == NoiseScheduler.EULER:
        scheduler = EulerDiscreteScheduler(beta_schedule="scaled_linear", prediction_type="v_prediction")
    elif scheduler == NoiseScheduler.EULER_A:
        scheduler = EulerAncestralDiscreteScheduler(beta_schedule="scaled_linear", prediction_type="v_prediction")
    elif scheduler == NoiseScheduler.DDIM:
        scheduler = DDIMScheduler(
            beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction", steps_offset=0, thresholding=True
        )
    elif scheduler == NoiseScheduler.DPM:
        scheduler = DPMSolverMultistepScheduler(beta_schedule="scaled_linear", prediction_type="v_prediction")
    elif scheduler == NoiseScheduler.LMS:
        scheduler = LMSDiscreteScheduler(beta_schedule="scaled_linear", prediction_type="v_prediction")
    elif scheduler == NoiseScheduler.PNDM:
        scheduler = PNDMScheduler(beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction", steps_offset=0)
    else:
        raise ValueError(f'Invalid scheduler: {scheduler}')

    return scheduler
