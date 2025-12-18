from typing import Union, List, TypeAlias, Tuple, TYPE_CHECKING
import numpy.typing as npt
import numpy as np
from typing import Optional

if TYPE_CHECKING:
    from uaibot.simobjects import (Ball, Box, ConvexPolytope, Cylinder, PointCloud, 
                                   Frame, RigidObject, Group, PointLight, Arrow, 
                                   HTMLDiv)
    from uaibot.robot import Robot
    from uaibot.graphics import MeshMaterial


Vector: TypeAlias = Union[List[float], Tuple[float], npt.NDArray[np.float64], np.matrix]
Matrix: TypeAlias = Union[npt.NDArray[np.float64], np.matrix]
HTMatrix: TypeAlias = np.matrix
MetricObject: TypeAlias = Union["Ball", "Box", "ConvexPolytope", "Cylinder", "PointCloud"]
GroupableObject: TypeAlias = Union["Ball", "Box", "Cylinder", "ConvexPolytope", "Frame",
                    "RigidObject", "Group", "Robot", "PointLight"]
SimObject: TypeAlias = Union["Ball", "Box", "Cylinder", "ConvexPolytope", "Frame",
                    "RigidObject", "Group", "Robot", "PointLight", "Arrow", "HTMLDiv"]


