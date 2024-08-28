import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from utils import COLORS

# GMM class definition
class GMM:
    def __init__(self, ncomp, mus, covs, priors):
        self.ncomp = ncomp
        self.mus = mus
        self.covs = covs
        self.priors = priors

    def inference(self, X):
        beliefs = np.zeros((X.shape[0], self.ncomp))
        for i in range(self.ncomp):
            rv = multivariate_normal(mean=self.mus[i], cov=self.covs[i])
            beliefs[:, i] = self.priors[i] * rv.pdf(X)

        log_likelihood = np.sum(np.log(np.sum(beliefs, axis=1)))
        beliefs /= np.sum(beliefs, axis=1, keepdims=True)
        return beliefs, log_likelihood

    def update(self, X, beliefs):
        Nk = np.sum(beliefs, axis=0)
        self.priors = Nk / X.shape[0]
        self.mus = np.dot(beliefs.T, X) / Nk[:, np.newaxis]
        self.covs = []

        for i in range(self.ncomp):
            X_shifted = X - self.mus[i]
            cov = np.dot(beliefs[:, i] * X_shifted.T, X_shifted) / Nk[i]
            self.covs.append(cov + 1e-6 * np.eye(X.shape[1]))  # Add regularization term

def segment_image(image_pixels, gmm, image_height, image_width, n_clusters):
    segmented_map = np.zeros((image_height, image_width, 3))
    binary_map = np.zeros((image_height, image_width))

    beliefs, _ = gmm.inference(image_pixels)

    for i in range(n_clusters):
        mask = np.argmax(beliefs, axis=1) == i
        segmented_map[mask.reshape(image_height, image_width)] = COLORS[i % len(COLORS)]
        binary_map[mask.reshape(image_height, image_width)] = 1 if i == 0 else binary_map[mask.reshape(image_height, image_width)]

    segmented_map = segmented_map.astype(np.uint8)
    binary_map = binary_map.astype(np.uint8)
    return segmented_map, binary_map

def segment_image_with_gmm(image_pixels, image_height, image_width, ncomp):
    # Normalize the pixels
    _mean = np.mean(image_pixels, axis=0, keepdims=True)
    _std = np.std(image_pixels, axis=0, keepdims=True)
    image_pixels = (image_pixels - _mean) / _std

    # Initialize using KMeans
    kmeans = KMeans(n_clusters=ncomp)
    labels = kmeans.fit_predict(image_pixels)
    initial_mus = kmeans.cluster_centers_
    initial_covs = [np.cov(image_pixels[labels == i].T) for i in range(ncomp)]
    initial_priors = [np.mean(labels == i) for i in range(ncomp)]

    # Create the GMM model
    gmm = GMM(ncomp, initial_mus, initial_covs, initial_priors)

    # Perform EM algorithm
    prev_log_likelihood = None
    for i in range(100):
        beliefs, log_likelihood = gmm.inference(image_pixels)  # E-step
        gmm.update(image_pixels, beliefs)  # M-step
        if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < 1e-10:
            break
        prev_log_likelihood = log_likelihood

    segmented_map, binary_map = segment_image(image_pixels, gmm, image_height, image_width, ncomp)
    return segmented_map, binary_map
