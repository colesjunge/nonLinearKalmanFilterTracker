import numpy as np
from fusion.sim.noise import contaminated_gaussian_noise

def test_shape():
    """
    Test whether the output of contaminated_gaussian_noise has correct shape relative to R
    """
    rng = np.random.default_rng(42)

    R1 = np.array([[4]])
    R2 = np.array([[4,0],
                  [0,4]])

    result_R1 = contaminated_gaussian_noise(rng=rng, R=R1)
    result_R2 = contaminated_gaussian_noise(rng=rng, R=R2)

    assert np.shape(R1[0]) == np.shape(result_R1), "Gaussian noise shape incorrect"
    assert np.shape(R2[0]) == np.shape(result_R2), "Gaussian noise shape incorrect"

def test_matches_mixture_covariance():
    """
    Test whether noise mixture matches theoretical covariance
    """
    rng = np.random.default_rng(42)
    epsilon = 0.1
    kappa = 10.0

    noise_list = []
    R = np.array([[4,0],
                [0,4]])

    for _ in range(100000):
        noise_list.append(contaminated_gaussian_noise(rng=rng, R=R))

    noise_list = np.array(noise_list)

    empirical_cov = np.cov(m=noise_list, rowvar=False, ddof=0)
    theoretical_cov = (1-epsilon)*R + epsilon*(kappa**2*R)

    assert np.allclose(empirical_cov, theoretical_cov, atol=2.0), "Empirical covariance does not converge to theoretical covariance"



def test_epsilon_zero_matches_nominal():
    """
    Test whether empirical covariance matches non contaminated covariance
    """

    rng = np.random.default_rng(42)
    epsilon = 0.0

    noise_list = []
    R = np.array([[4,0],
                [0,4]])

    for _ in range(100000):
        noise_list.append(contaminated_gaussian_noise(rng=rng, epsilon=epsilon, R=R))

    noise_list = np.array(noise_list)

    empirical_cov = np.cov(m=noise_list, rowvar=False, ddof=0)
    theoretical_cov = R

    assert np.allclose(empirical_cov, theoretical_cov, atol=2.0), "Empirical covariance does not converge to theoretical covariance"

def test_epsilon_one_matches_contaminated():
    """
    Test whether empirical covariance matches contaminated covariance
    """
    rng = np.random.default_rng(42)
    kappa = 10.0
    epsilon = 1.0

    noise_list = []
    R = np.array([[4,0],
                [0,4]])
    
    for _ in range(100000):
        noise_list.append(contaminated_gaussian_noise(rng=rng, epsilon=epsilon, R=R))

    noise_list = np.array(noise_list)

    empirical_cov = np.cov(m=noise_list, rowvar=False, ddof=0)
    theoretical_cov = kappa**2*R

    assert np.allclose(empirical_cov, theoretical_cov, atol=2.0), "Empirical covariance does not converge to theoretical covariance"


def test_reproducible_with_seed():
    """
    Test whether same seed produces same results
    """

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    R = np.array([[4,0],
                [0,4]])
    
    result_rng1 = contaminated_gaussian_noise(rng=rng1, R=R)
    result_rng2 = contaminated_gaussian_noise(rng=rng2, R=R)

    assert np.array_equal(result_rng1, result_rng2), "Same seed does not produce same results"

# To run: pytest tests/test_contaminated_noise.py -v