import numpy as np
import pytest
from uaibot.utils import Utils

class TestTypeChecks:
    # Test is_a_number
    def test_is_a_number(self):
        assert Utils.is_a_number(5) is True
        assert Utils.is_a_number(3.14) is True
        assert Utils.is_a_number(np.float64(2.5)) is True
        assert Utils.is_a_number(np.float64(2)) is True
        assert Utils.is_a_number("5") is False
        assert Utils.is_a_number([1]) is False
        assert Utils.is_a_number(None) is False
        assert Utils.is_a_number(np.array([1])) is False
        # assert Utils.is_a_number(np.nan) is False
        # assert Utils.is_a_number(np.inf) is False

    # Test is_a_natural_number
    def test_is_a_natural_number(self):
        assert Utils.is_a_natural_number(5) is True
        assert Utils.is_a_natural_number(0) is True
        assert Utils.is_a_natural_number(-1) is False
        assert Utils.is_a_natural_number(3.14) is False
        assert Utils.is_a_natural_number("5") is False
        assert Utils.is_a_natural_number([1]) is False
        assert Utils.is_a_natural_number(None) is False
        assert Utils.is_a_natural_number(np.array([1])) is False
        assert Utils.is_a_natural_number(np.nan) is False
        assert Utils.is_a_natural_number(np.inf) is False
        assert Utils.is_a_natural_number(np.float64(5)) is False

    # Test is_a_matrix
    def test_is_a_matrix(self):
        # Valid cases
        assert Utils.is_a_matrix(np.array([[1, 2], [3, 4]])) is True
        assert Utils.is_a_matrix([[1, 2], [3, 4]]) is True
        assert Utils.is_a_matrix(np.matrix([[1, 2], [3, 4]])) is True
        
        # Size checks
        assert Utils.is_a_matrix([[1, 2], [3, 4]], n=2, m=2) is True
        assert Utils.is_a_matrix([[1, 2], [3, 4]], n=3, m=3) is False
        assert Utils.is_a_matrix([[1, 2], [3, 4]], n=1, m=2) is False
        assert Utils.is_a_matrix([[1, 2], [3, 4]], n=3, m=None) is False
        assert Utils.is_a_matrix([[1, 2], [3, 4]], n=2, m=3) is False
        assert Utils.is_a_matrix([[1, 2], [3, 4]], n=None, m=3) is False
        
        # Invalid cases
        assert Utils.is_a_matrix([[1, "a"], [3, 4]]) is False
        # assert Utils.is_a_matrix("not a matrix") is False # this returns nothing??
        # assert Utils.is_a_matrix(None) is False # this returns nothing??

    # Test is_a_vector
    def test_is_a_vector(self):
        # Valid cases
        assert Utils.is_a_vector(np.array([1, 2, 3])) is True
        assert Utils.is_a_vector(np.array([1, 2, 3]), n=3) is True
        assert Utils.is_a_vector([[1], [2], [3]]) is True
        assert Utils.is_a_vector([[1], [2], [3]], n=3) is True
        
        # Size checks
        assert Utils.is_a_vector([1, 2, 3], n=0) is False
        assert Utils.is_a_vector([1, 2, 3], n=2) is False
        assert Utils.is_a_vector([[1], [2], [3]], n=2) is False
        
        # Invalid cases
        assert Utils.is_a_vector([[1, 2], [3, 4]]) is False
        assert Utils.is_a_vector(np.array([[1, 2], [3, 4]])) is False
        assert Utils.is_a_vector(np.matrix([[1, 2], [3, 4]])) is False
        # assert Utils.is_a_vector(1) is False # this returns nothing??
        # assert Utils.is_a_vector(1.0) is False # this returns nothing??
        # assert Utils.is_a_vector(None) is False # this returns nothing??

    # Test is_a_pd_matrix
    def test_is_a_pd_matrix(self):
        # Valid PD matrix
        M0 = np.array([[2, -1], [-1, 2]])
        assert Utils.is_a_pd_matrix(M0) is True
        
        M = [[2, -1], [-1, 2]]
        assert Utils.is_a_pd_matrix(M) is True

        M = np.matrix([[2, -1], [-1, 2]])
        assert Utils.is_a_pd_matrix(M) is True

        M = np.array([[1, 1], [1, 1]])
        assert Utils.is_a_pd_matrix(M) is False
        
        M = np.array([[1, 2], [0, 1]])
        assert Utils.is_a_pd_matrix(M) is False
        
        # Size check
        assert Utils.is_a_pd_matrix(M0, n=2) is True
        assert Utils.is_a_pd_matrix(M0, n=3) is False

        # Not matrices
        # assert Utils.is_a_pd_matrix(1) is False
        # assert Utils.is_a_pd_matrix(1.0) is False
        # assert Utils.is_a_pd_matrix(None) is False