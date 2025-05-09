import numpy as np
import pytest
from src.moment_utils import moments, moment_eq, calc_moment


def test_moments_basic():
    """Test moments function with basic inputs"""
    beta = 1.0
    w = 0.0
    ci = -1.0
    cf = 1.0
    
    result = moments(beta, w, ci, cf)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    
    # Test symmetry around w=0
    result_pos = moments(beta, 0.5, ci, cf)
    result_neg = moments(beta, -0.5, ci, cf)
    assert np.isclose(result_pos[0], -result_neg[0])
    assert np.isclose(result_pos[1], result_neg[1])

def test_moments_array_input():
    """Test moments function with array inputs"""
    beta = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([[0.0, 0.5], [-0.5, 0.0]])
    ci = -1.0
    cf = 1.0
    
    result = moments(beta, w, ci, cf)
    assert result[0].shape == beta.shape
    assert result[1].shape == beta.shape

def test_moments_edge_cases():
    """Test moments function with edge cases"""
    # Test with very small beta
    result = moments(1e-10, 0.0, -1.0, 1.0)
    assert not np.isnan(result[0])
    assert not np.isnan(result[1])
    
    # Test with very large beta
    result = moments(1e10, 0.0, -1.0, 1.0)
    assert not np.isnan(result[0])
    assert not np.isnan(result[1])

def test_moment_eq_basic():
    """Test moment_eq function with basic inputs"""
    x = np.array([1.0, 0.0])  # [beta, w]
    u = 0.0  # target velocity
    e = 1.0  # target energy
    ci = -1.0
    cf = 1.0
    
    result = moment_eq(x, u, e, ci, cf)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)

def test_moment_eq_consistency():
    """Test consistency between moments and moment_eq"""
    beta = 1.0
    w = 0.0
    ci = -1.0
    cf = 1.0
    
    # Get moments
    u, e = moments(beta, w, ci, cf)
    
    # Check if moment_eq returns close to zero for these parameters
    result = moment_eq(np.array([beta, w]), u, e, ci, cf)
    assert np.allclose(result, [0.0, 0.0], atol=1e-10)

def test_calc_moment_basic():
    """Test calc_moment function with basic inputs"""
    # Create a simple Gaussian distribution
    cx_vec = np.linspace(-3, 3, 10)
    cy_vec = np.linspace(-3, 3, 10)
    cz_vec = np.linspace(-3, 3, 10)
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')
    
    # Create a normalized Gaussian
    f = np.exp(-(cx**2 + cy**2 + cz**2)) / (np.pi)**1.5
    
    result = calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec)
    assert len(result) == 3
    assert np.isclose(result[0], 1.0, atol=1e-2)  # density should be 1
    assert np.allclose(result[1], 0.0, atol=1e-2)  # momentum should be 0
    assert np.isclose(result[2], 1.5, atol=1e-2)  # energy should be 1.5

def test_calc_moment_zero():
    """Test calc_moment with zero distribution"""
    cx_vec = np.linspace(-3, 3, 10)
    cy_vec = np.linspace(-3, 3, 10)
    cz_vec = np.linspace(-3, 3, 10)
    cx, cy, cz = np.meshgrid(cx_vec, cy_vec, cz_vec, indexing='ij')
    
    f = np.zeros_like(cx)
    result = calc_moment(f, cx, cy, cz, cx_vec, cy_vec, cz_vec)
    assert np.allclose(result, 0.0)

if __name__ == '__main__':
    pytest.main([__file__]) 