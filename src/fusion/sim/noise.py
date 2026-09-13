import numpy as np

def contaminated_gaussian_noise(rng: np.random.Generator,
                                R: np.ndarray,
                                epsilon: float = 0.1,
                                kappa: float = 10.0) -> np.ndarray:
    """Draw noise using multivariate normal distribution and inject Huber style loss
    In: R (dim_z, dim_z), Sensor's nominal covariance; epsilon float, prob of contaminated noise; kappa scales gaussian variation (contamination)
    Out: total_noise, Single noise draw (dim_z, )           
    """

    # Generate noise and contaminated noise
    mean = np.zeros(len(R))
    noise = rng.multivariate_normal(mean=mean, cov=R, method='cholesky')
    contaminated_noise = rng.multivariate_normal(mean=mean, cov=(kappa**2)*R, method='cholesky') # Huber style loss

    # Choose contaminated with prob epsilon
    contaminated = rng.binomial(n=1, p=epsilon)

    if contaminated:
        final_noise = contaminated_noise
    else:
        final_noise = noise

    return final_noise
