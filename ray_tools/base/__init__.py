from .backend import RayOutput
from .transform import RayTransform

RayTransformType = RayTransform | dict[str, RayTransform]

