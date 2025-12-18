import numpy as np
import pytest
from math import acos, sqrt, atan2, sin, cos
from uaibot.utils import Utils

class TestMathFunctions:
    # Test S (cross product matrix)
    def test_S_matrix(self):
        v = [1, 2, 3]
        S = Utils.S(v)
        expected = np.array([[0, -3, 2],
                            [3, 0, -1],
                            [-2, 1, 0]])
        assert np.allclose(S, expected, atol=1e-10)
        
        v = np.array([1, 2, 3])
        S = Utils.S(v)
        assert np.allclose(S, expected, atol=1e-10)

        v = np.matrix([1, 2, 3])
        S = Utils.S(v)
        assert np.allclose(S, expected, atol=1e-10)

    # Test rotation around arbitrary axis
    def test_rot(self):
        axis = [1, 0, 0]
        angle = np.pi/2
        R = Utils.rot(axis, angle)
        expected = Utils.rotx(angle)
        assert np.allclose(R, expected, atol=1e-10)

        # Test orthogonality
        assert np.allclose(R[:3,:3].T @ R[:3,:3], np.eye(3))

        axis = [0, 1, 0]
        R = Utils.rot(axis, angle)
        expected = Utils.roty(angle)
        assert np.allclose(R, expected, atol=1e-10)

        # Test orthogonality
        assert np.allclose(R[:3,:3].T @ R[:3,:3], np.eye(3))

        axis = [0, 0, 1]
        R = Utils.rot(axis, angle)
        expected = Utils.rotz(angle)
        assert np.allclose(R, expected, atol=1e-10)

        # Test orthogonality
        assert np.allclose(R[:3,:3].T @ R[:3,:3], np.eye(3), atol=1e-10)

        # Test with different input types
        axis = np.array([1, 0, 0])
        R = Utils.rot(axis, angle)
        expected = Utils.rotx(angle)
        assert np.allclose(R, expected, atol=1e-10)

        axis = np.matrix([1, 0, 0])
        R = Utils.rot(axis, angle)
        expected = Utils.rotx(angle)
        assert np.allclose(R, expected, atol=1e-10)

    # Test translation matrix
    def test_trn(self):
        v = [1, 2, 3]
        T = Utils.trn(v)
        expected = np.array([[1, 0, 0, 1],
                            [0, 1, 0, 2],
                            [0, 0, 1, 3],
                            [0, 0, 0, 1]])
        assert np.allclose(T, expected, atol=1e-10)

        # Check different input types
        v = np.array([1, 2, 3])
        T = Utils.trn(v)
        expected = np.array([[1, 0, 0, 1],
                            [0, 1, 0, 2],
                            [0, 0, 1, 3],
                            [0, 0, 0, 1]])
        assert np.allclose(T, expected, atol=1e-10)

        v = np.matrix([1, 2, 3])
        T = Utils.trn(v)
        expected = np.array([[1, 0, 0, 1],
                            [0, 1, 0, 2],
                            [0, 0, 1, 3],
                            [0, 0, 0, 1]])
        assert np.allclose(T, expected, atol=1e-10)

    # Test axis-specific rotations
    def test_rotx(self):
        angle = np.pi/4
        R = Utils.rotx(angle)
        expected = np.array([[1, 0, 0, 0],
                            [0, cos(angle), -sin(angle), 0],
                            [0, sin(angle), cos(angle), 0],
                            [0, 0, 0, 1]])
        assert np.allclose(R, expected, atol=1e-10)

    def test_roty(self):
        angle = np.pi/3
        R = Utils.roty(angle)
        expected = np.array([[cos(angle), 0, sin(angle), 0],
                            [0, 1, 0, 0],
                            [-sin(angle), 0, cos(angle), 0],
                            [0, 0, 0, 1]])
        assert np.allclose(R, expected, atol=1e-10)

    def test_rotz(self):
        angle = np.pi/6
        R = Utils.rotz(angle)
        expected = np.array([[cos(angle), -sin(angle), 0, 0],
                            [sin(angle), cos(angle), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        assert np.allclose(R, expected, atol=1e-10)

    # Test random HTM generation
    def test_htm_rand(self):
        np.random.seed(42)  # For reproducible tests
        htm = Utils.htm_rand()
        
        # Check basic structure
        assert htm.shape == (4, 4)
        assert np.allclose(htm[3,:], [0, 0, 0, 1])
        
        # Check rotation matrix properties
        R = htm[:3,:3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) > 0.99  # Should be close to 1

    # Test HTM inversion
    def test_inv_htm(self):
        htm = Utils.rotx(np.pi/4) @ Utils.trn([1, 2, 3])
        inv_htm = Utils.inv_htm(htm)
        
        # Should be exact inverse
        assert np.allclose(htm @ inv_htm, np.eye(4), atol=1e-10)
        assert np.allclose(inv_htm @ htm, np.eye(4), atol=1e-10)
        
        # Check specific properties
        R = htm[:3,:3]
        p = htm[:3,3]
        expected_inv = np.eye(4)
        expected_inv[:3,:3] = R.T
        expected_inv[:3,3] = np.array(-R.T @ p).ravel()
        assert np.allclose(inv_htm, expected_inv, atol=1e-10)

    # Test axis-angle extraction
    def test_axis_angle(self):
        axis = np.array([1, 0, 0])
        angle = np.pi/3
        htm = Utils.rot(axis, angle)
        extracted_axis, extracted_angle = Utils.axis_angle(htm)
        
        # Check angle
        assert bool(np.isclose(extracted_angle, angle, atol=1e-8))
        
        # Check axis (direction only, sign may flip)
        assert bool(np.isclose(np.abs((extracted_axis.T @ axis).item()), 1, atol=1e-8))
        
        # Test edge case (zero rotation)
        # NEEDS TO BE FIXED -- HANGS
        # htm = np.eye(4)
        # axis, angle = Utils.axis_angle(htm)
        # assert angle == 0

    # Test Euler angles extraction
    def test_euler_angles(self):
        # Test standard case
        alpha, beta, gamma = np.pi/4, np.pi/6, np.pi/3
        htm = Utils.rotz(alpha) @ Utils.roty(beta) @ Utils.rotx(gamma)
        a, b, c = Utils.euler_angles(htm)
        
        assert np.isclose(a, alpha, atol=1e-8)
        assert np.isclose(b, beta, atol=1e-8)
        assert np.isclose(c, gamma, atol=1e-8)
        
        # Test gimbal lock case (beta ≈ 0)
        htm = Utils.rotz(0.5) @ Utils.rotx(0.3)  # beta=0
        a, b, c = Utils.euler_angles(htm)
        reconstructed = Utils.rotz(a) @ Utils.roty(b) @ Utils.rotx(c)
        assert np.allclose(htm, reconstructed, atol=1e-8)

