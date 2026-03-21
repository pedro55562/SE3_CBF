import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
import os
from setup import *
import itertools
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
from scipy.linalg import expm

robot = ub.Robot.create_rigid_body_se3()
jg , fkm = robot.jac_geo()
print(jg)