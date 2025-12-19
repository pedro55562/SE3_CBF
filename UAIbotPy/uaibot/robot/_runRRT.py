from utils import *
import numpy as np
import math
import os
import uaibot as ub
from typing import Optional, List, Tuple


def _runRRT(
    self,
    q_start: Optional[Vector] = None,
    htm: Optional[HTMatrix] = None,
    q_goal: Optional[List[Vector]] = None,
    htm_target: Optional[HTMatrix] = None,
    obstacles: List[MetricObject] = [],
    max_iter: int = 2000,
    n_tries: int = 10,
    goal_tolerance: float = 0.15,
    goal_bias: float = 0.35,
    step_size_min: float = 0.2,
    step_size_max: float = 1.5,
    usemultthread: bool = True,
) -> Tuple[bool, list, int, int, float]:

    # Run RRT planner for the robot. Returns (success, path, iterations, num_tries, planning_time)

    if q_start is None:
        q_start = self.q
    if htm is None:
        htm = self.htm

    # === Error handling  ===
    n = len(self._links)
    if not Utils.is_a_vector(q_start, n):
        raise Exception("The parameter 'q_start' should be a" + str(n) +"dimensional vector.")

    if not Utils.is_a_natural_number(max_iter):
        raise Exception("The parameter 'max_iter' should be a nonnegative integer number.")

    if not Utils.is_a_natural_number(n_tries):
        raise Exception("The parameter 'n_tries' should be a nonnegative integer number.")

    if (not Utils.is_a_number(goal_tolerance)) or goal_tolerance <= 0:
        raise Exception("The parameter 'goal_tolerance' should be a positive number.")

    if (not Utils.is_a_number(goal_bias)) or (goal_bias < 0 or goal_bias > 1):
        raise Exception("The parameter 'goal_bias' should be a number in [0,1].")

    if (not Utils.is_a_number(step_size_min)) or step_size_min <= 0:
        raise Exception("The parameter 'step_size_min' should be a positive number.")
    
    if (not Utils.is_a_number(step_size_max)) or step_size_max <= 0:
        raise Exception("The parameter 'step_size_max' should be a positive number.")
    
    if not str(type(usemultthread)) == "<class 'bool'>":
        raise Exception("The parameter 'usemultthread' should be a boolean.")
    
    for obs in obstacles:
        if not Utils.is_a_metric_object(obs):
            raise Exception("The parameter 'obstacles' should be a list of metric objects : " + str(Utils.IS_METRIC) + ".")
    try:
        ok_init, _, _ = self.check_free_config(Utils.cvt(q_start), htm, obstacles, True, True, mode='auto')
        if not ok_init:
            raise Exception("Initial configuration 'q_start' is in collision. Provide a collision-free start configuration.")
    except AttributeError:
        raise Exception("Robot does not expose 'check_free_config' required to validate the initial configuration.")
    
    if q_goal is not None:
        q_goal_list = q_goal if isinstance(q_goal, (list, tuple)) else [q_goal]
        for idx, qg in enumerate(q_goal_list):
            if not Utils.is_a_vector(qg, n):
                raise Exception("q_goal[%d] must be a vector of length %d." % (idx, n))
            try:
                ok_goal, _, _ = self.check_free_config(Utils.cvt(qg), htm, obstacles, True, True, mode='auto')
                if not ok_goal:
                    raise Exception("q_goal[" + str(idx) + "] is in collision. Provide only collision-free goal configurations.")
            except AttributeError:
                raise Exception("Robot does not expose 'check_free_config' required to validate q_goal configurations.")
    
    if htm_target is not None:
        if not Utils.is_a_matrix(htm_target, 4, 4):
            raise Exception("The parameter 'htm_target' should be a 4x4 homogeneous transformation matrix or None.")

        ik_ok = False
        pretries = min(3, n_tries)
        for _try in range(pretries):
            try:
                q_test = self.ikm(htm_tg=htm_target, htm=htm, obstacles=obstacles, no_tries = 200, no_iter_max=400)
                ik_ok = True
                break
            except Exception:
                pass

        if not ik_ok:
            raise Exception("Quick IK pre-check failed for 'htm_target'. Try increasing 'no_tries'/'max_iter' or loosening tolerances.")
 
 
    # Convert obstacles to C++ objects when needed
    obstacles_cpp = []
    for obs in obstacles:
        if Utils.get_uaibot_type(obs) == 'uaibot.CPP_GeometricPrimitives':
            obstacles_cpp.append(obs)
        else:
            obstacles_cpp.append(obs.cpp_obj)

    planning_time = 0
    iterations = 0
    i = 0

    while i < n_tries:
        i += 1
        q_goal_cpp = []

        # If explicit goal joint configurations were provided, use them
        if (htm_target is None) and (q_goal is not None):
            q_goal_cpp = q_goal

        # Otherwise, try to generate IK solutions for the target transform
        if (q_goal is None) and (htm_target is not None):
            for _ in range(10):
                try:
                    q_goal_cpp.append(
                        self.ikm(htm_tg=htm_target, htm=htm, obstacles=obstacles)
                    )
                except Exception:
                    # ignore failed IK attempts and try again
                    pass

        # Remove near-duplicate IK solutions to avoid redundant goals
        unique_q_goal = []
        for q in q_goal_cpp:
            if not any(
                np.linalg.norm(np.array(q) - np.array(q2)) < 0.3 for q2 in unique_q_goal
            ):
                unique_q_goal.append(q)

        q_goal_cpp = unique_q_goal

        # Create and run the C++ RRT planner
        cpp_robot = self.cpp_robot
        rrt_instance = ub_cpp.CPP_RRT(
            cpp_robot,
            Utils.cvt(q_start),
            [Utils.cvt(q) for q in q_goal_cpp],
            htm,
            obstacles_cpp,
            max_iter,
            goal_tolerance,
            goal_bias,
            step_size_min,
            step_size_max,
            usemultthread,
        )

        result = rrt_instance.runRRT()
        planning_time += result.planning_time
        iterations += result.iterations

        if result.success:
            num_tries = i
            break

    return (
        result.success,
        [Utils.cvt(q) for q in result.path],
        iterations,
        num_tries,
        planning_time,
    )
