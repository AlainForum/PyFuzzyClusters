import numpy as np
import abc

from scipy.spatial import distance


class Distance(object):

    __metaclass__ = abc.ABCMeta

    def vector_difference(self, x, y=None):
        y = self.check_y(x, y)
        return x - y[:, np.newaxis]


    @staticmethod
    def check_y(x, y):
        if y is None:
            return np.copy(x)
        else:
            return y

    @abc.abstractmethod
    def __call__(self, x, y=None, metric='euclidean', **kwargs):
        if y is None:
            return distance.squareform(distance.pdist(x, metric, **kwargs))
        else:
            return distance.cdist(x, y, metric, **kwargs).T


class Euclidean(Distance):

    def __call__(self, x, y=None, **kwargs):
        """Euclidean distance metric.

        If one argument is passed, calculates the internal distance matrix.

        If two arguments are passed, calculates the Euclidean lengths of the
        difference vectors from every vector in the first to every vector in the
        second.

        Parameters
        ----------
        x : ndarray
            (n_vectors, n_dimensions)
            An array of vectors
        y : Optional [ndarray]
            (m_vectors, n_dimensions)
            Optional second array of vectors.

        Returns
        -------
        ndarray
            (n_vectors, m_vectors)
            The Euclidean distance between vector n and vector m.

        """
        return super(Euclidean, self).__call__(x, y, metric='euclidean',
                                               **kwargs)


class IMED(Distance):

    def __init__(self, image_shape, width_factor=1.):
        self.image_shape = image_shape
        self.pixel_distances = self.calculate_pixel_distances(width_factor)
        super(IMED, self).__init__()

    def calculate_pixel_distances(self, width_factor):
        nx, ny = self.image_shape
        xs = np.arange(0, nx, 1)
        ys = np.arange(0, ny, 1)
        px, py = np.meshgrid(xs, ys)
        px = px.flatten()
        py = py.flatten()
        dp = np.exp((-np.add(np.square(px - px[:, np.newaxis]),
                             np.square(py - py[:, np.newaxis]))) / (
                    2 * width_factor))
        return dp

    def __call__(self, x, y=None):
        """Calculates the IMage Euclidean Distance between two images.

        If one argument is passed, calculates the internal distance matrix.

        If two arguments are passed, calculates the IMED lengths of the
        difference vectors from every vector in the first to every vector in the
        second.

        Parameters
        ----------
        x : ndarray
            (n_images, image_dim)
            Matrix of image intensities.
        y : Optional [ndarray]
            (m_images, image_dim)
            Matrix of image intensities.

        Returns
        -------
        d : ndarray
            The image euclidean distance.

        See Also
        --------
        calculate_pixel_distances : Calculation of image pixel distances.

        """
        v = super(IMED, self).__call__(x, y)
        d = np.sum(np.dot(v, self.pixel_distances) * v, axis=-1)
        return d


class Correlation(Distance):

    def __call__(self, x, y=None, **kwargs):
        return super(Correlation, self).__call__(x, y, metric='correlation',
                                                 **kwargs)


class Cosine(Distance):
    
    def __call__(self, x, y=None, **kwargs):
        return super(Cosine, self).__call__(x, y, metric='cosine', **kwargs)


class Canberra(Distance):

    def __call__(self, x, y=None, **kwargs):
        return super(Canberra, self).__call__(x, y, metric='canberra', **kwargs)
