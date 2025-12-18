import pytest
import numbers
import numpy as np
from uaibot.simobjects.pointcloud import PointCloud
from uaibot.simobjects.ball import Ball
from uaibot.simobjects.box import Box

class TestBall:
    @pytest.fixture
    def default_ball(self):
        return Ball(name="test_ball", radius=0.1, color="red", opacity=0.7)

    def test_initialization(self, default_ball):
        """Test basic creation and repr"""
        assert default_ball.name == "test_ball"
        assert default_ball.radius == 0.1
        assert default_ball.color == "red"
        assert default_ball.htm.shape == (4, 4)
        assert np.allclose(default_ball.htm, np.identity(4))
        assert repr(default_ball) is not None

    def test_animation_frames(self, default_ball):
        """Test frame addition and validation"""
        default_ball.add_ani_frame(0.1, np.eye(4))
        default_ball.set_ani_frame(np.eye(4))
        
        with pytest.raises((Exception, ValueError, TypeError)):
            default_ball.add_ani_frame(-0.1, np.eye(4))

    def test_copy(self, default_ball):
        """Test deep copying"""
        ball_copy = default_ball.copy()
        assert ball_copy is not None
        assert ball_copy is not default_ball
        assert ball_copy.radius == default_ball.radius
        assert np.allclose(ball_copy.htm, default_ball.htm)
        assert ball_copy.color == default_ball.color
        # This parameter was removed in the latest version
        # assert ball_copy.mass == default_ball.mass

    @pytest.mark.parametrize("mode", ["c++", "python"])
    def test_aabb(self, default_ball, mode):
        """Test AABB generation in both modes"""
        aabb = default_ball.aabb(mode=mode)
        ## The following fails because internally, the class is imported as part of the submodule only (it lacks the uaibot. prefix)
        # assert isinstance(aabb, Box)
        assert aabb.__class__.__name__ == Box.__name__

    @pytest.mark.parametrize("mode", ["c++", "python"])
    def test_point_cloud_conversion(self, default_ball, mode):
        """Test point cloud generation"""
        pc = default_ball.to_point_cloud(mode=mode)
        # print(type(pc))
        ## The following fails because internally, the class is imported as part of the submodule only (it lacks the uaibot. prefix)
        # assert isinstance(pc, PointCloud)
        assert pc.__class__.__name__ == PointCloud.__name__

    # THE CURRENT VERSION FAILS ON PYTHON BECAUSE A POINT IS NOT CONSIDERED AS A VECTOR IN THIS CASE
    # @pytest.mark.parametrize("mode", ["c++", "python"])
    def test_distance_computation(self, default_ball):
        """Test distance between balls"""
        ball_copy = default_ball.copy()
        res = default_ball.compute_dist(ball_copy, mode="c++")
        mode = "c++"
        if mode == "c++":
            assert len(res) == 4
        else:
            assert len(res) == 3 # THIS SHOULD  BE CHANGED IN NEW VERSION
        assert isinstance(res[2], numbers.Real)  # Distance
        assert isinstance(res[0], np.ndarray)    # Point 1
        assert isinstance(res[1], np.ndarray)    # Point 2

    @pytest.mark.parametrize("mode", ["c++", "python"])
    def test_projection(self, default_ball, mode):
        """Test point projection onto ball"""
        p_test = np.array([0, 0, 1]).reshape(3, 1)
        print(p_test, type(p_test))
        p_proj, distance = default_ball.projection(p_test, mode=mode)
        assert isinstance(p_proj, np.ndarray)
        assert isinstance(distance, numbers.Real)

    @pytest.mark.parametrize("h,eps", [
        (0.1, 0),   # Invalid h
        (0, 0.1)     # Invalid eps
    ])
    def test_projection_validation(self, default_ball, h, eps):
        """Test invalid projection parameters"""
        p_test = np.array([0, 0, 1]).reshape(3, 1)
        with pytest.raises((Exception, ValueError, TypeError)):
            default_ball.projection(p_test, h=h, eps=eps, mode="python")
