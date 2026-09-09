from matplotlib.patches import Ellipse
import numpy as np


def plot_covariance_ellipse(ax, mean, cov, n_std=2, **kwargs):
    """
    Plot an n-standard-deviation covariance ellipse.

    In:
        ax; Matplotlib Axes to add the ellipse to.
        mean; (2,) array containing the ellipse center.
        cov; (2, 2) covariance matrix.
        n_std; Number of standard deviations represented by the ellipse.
        **kwargs; Additional arguments forwarded to matplotlib.patches.Ellipse.

    Out: ellipse; The created Ellipse patch.
    """

    # Eigenvalues/eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Choose the Eigenvector corresponding to the largest eigenvalue as it defines the major-axis direction.
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # The angle of the major axis in degrees
    angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

    # Standard deviations along the principal axes determine width adn height
    width = 2 * n_std * np.sqrt(eigenvalues[1])
    height = 2 * n_std * np.sqrt(eigenvalues[0])

    # Create and add ellipse
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        **kwargs
    )

    ax.add_patch(ellipse)

    return ellipse