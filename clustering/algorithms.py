import numpy as np
import abc
import distance


class AlternatingOptimization:

    __metaclass__ = abc.ABCMeta

    def __init__(self, x, n_clusters=2, max_iter=200, tol=1e-4,
                 distance_metric=distance.Euclidean(), **kwargs):
        """Alternating optimization (A/O) algorithm for clustering.

        Parameters
        ----------
        x : ndarray
            (n_samples, n_variables)
            Unlabeled object data.
        n_clusters : int
            Number of clusters to find.
        max_iter : int
            The limit on the number of iterations.
        tol : float
            The permitted accuracy in `j`.
        distance_metric : function
            The metric used to calculate the space.
        kwargs : Arguments passed to subclasses

        """

        # Store unlabeled data
        self.X = x

        # Set parameters
        self.Q = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.distance_metric = distance_metric
        self.params = kwargs

        # Set initial values
        self.D = None  # Distance matrix
        self.C = guess_prototypes(x, n_clusters)  # Cluster centers
        self.W = 0  # Cluster weights
        self.U = None  # Cluster memberships
        self.J = np.inf
        self.started = False

    def distance(self, x, c):
        """Calculates the distance metric between data 'x' and centers 'c'.

        Parameters
        ----------
        x : ndarray
            (n_samples, n_variables)
            Unlabeled object data.
        c : ndarray
            (n_clusters, n_variables)
            Cluster centers.

        Returns
        -------
        d : ndarray
            (n_clusters, n_samples)
            Distance metric.

        """
        return self.distance_metric(x, c)

    @abc.abstractmethod
    def u(self, d):
        """Calculates cluster membership based on distance to centers 'd'.

        Parameters
        ----------
        d : ndarray
            (n_clusters, n_samples)
            Distance metric.

        Returns
        -------
        u : ndarray
            (n_clusters, n_samples)
            Data point memberships.

        """
        return

    @abc.abstractmethod
    def c(self, x, u):
        """Calculates cluster centers based on memberships 'u'.

        Parameters
        ----------
        x : ndarray
            (n_samples, n_variables)
            Unlabeled object data.
        u : ndarray
            (n_clusters, n_samples)
            Data point memberships.

        Returns
        -------
        c : ndarray
            (n_clusters, n_variables)
            Cluster centers.

        """
        return

    @abc.abstractmethod
    def j(self, u, d):
        """Calculates the objective function based on memberships 'u'.

        Parameters
        ----------
        u : ndarray
            (n_clusters, n_samples)
            Data point memberships.
        d : ndarray
            (n_clusters, n_samples)
            Distance metric.

        Returns
        -------
        j : float
            Value of the objective function for this partition.

        """
        return

    def start(self):
        """First routine to be implemented by the subclasses before iteration.

        """
        pass

    def __iter__(self):
        return self

    def next(self):
        """Update distances, memberships, and cluster centers.

        Returns
        -------
        self.J : The new value of the minimisation function.

        """
        if self.started is False:
            self.start()
            self.started = True
        self.D = self.distance(self.X, self.C)
        self.U = self.u(self.D)  # Update memberships with the old centers
        self.C = self.c(self.X, self.U)  # Update centers with new memberships
        self.J = self.j(self.U, self.D)
        return self.J

    def optimize(self):
        for i in range(self.max_iter):
            if self.J < self.next() + self.tol:
                break
        return self

    @property
    def model(self):
        return np.dot(self.U.T, self.C)

    @property
    def entropy(self):
        return -np.sum(self.U * np.log(self.U))


class CMeans(AlternatingOptimization):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(CMeans, self).__init__(*args, **kwargs)
        if 'm' in kwargs:
            self.m = kwargs['m']
        else:
            self.m = 2

    def e(self, x, c):
        """Covariance of data assigned to each cluster

        Parameters
        ----------
        x : (n_samples, n_variables) array_like
            Unlabeled object data.
        c : (n_clusters, n_variables) array_like
            Cluster centers.

        Returns
        -------
        ndarray
            The (modified) covariance matrix of the data assigned to each
            cluster.

        """
        q, p = c.shape
        if self.U is None:
            return (np.eye(p)[..., np.newaxis] * np.ones((p, q))).T
        v = distance.Distance().vector_difference(x, c)
        u = self.g(self.U)
        outer = np.einsum('...i,...j->...ij', v, v)
        es = np.einsum('...i,...ijk', u, outer) / np.sum(u, axis=1)[
            ..., np.newaxis, np.newaxis]
        p = x.shape[1]
        return es / (np.linalg.det(es) ** (1. / p))[..., np.newaxis, np.newaxis]

    def g(self, u):
        """Fuzzification operator.

        Parameters
        ----------
        u : (n_clusters, n_samples) array_like
            Cluster memberships.

        Returns
        -------
        g : ndarray
            (n_clusters, n_samples)
            Fuzzified memberships.

        """
        g = u ** self.m
        return g


class Hard(CMeans):
    def u(self, d):
        p = d.shape[0]
        u = np.arange(p)[:, np.newaxis] == np.argmin(d, axis=0)
        return u

    def c(self, x, u):
        return np.dot(u, x) / np.sum(u, axis=1)[..., np.newaxis]

    def j(self, u, d):
        return np.sum(u * d)


class PossibilisticFuzzy(CMeans):
    def start(self):
        initializer = ProbabilisticFuzzy(
            self.X, self.Q, self.max_iter, self.tol, **self.params)
        initializer.optimize()
        self.D = initializer.D
        self.U = initializer.U
        self.C = initializer.C
        self.W = self._guess_weights(self.U, self.D)

    def _guess_weights(self, u, d):
        w = np.sum(self.g(u) * d, axis=1) / np.sum(self.g(u), axis=1)
        return w

    def u(self, d):
        return (1. + (d / self.W[:, np.newaxis]) ** (1. / (self.m - 1))) ** -1.

    def c(self, x, u):
        return np.dot(self.g(u), x) / np.sum(self.g(u), axis=1)[..., np.newaxis]

    def j(self, u, d):
        return np.sum(self.g(u) * d) \
               + np.sum(self.W * np.sum(self.g(1. - u), axis=1))


class ProbabilisticFuzzy(CMeans):

    def u(self, d):
        return 1/np.sum(np.divide(d, d[:, np.newaxis])**(2/(self.m-1)),
                        axis=0)

    def c(self, x, u):
        return np.dot(self.g(u), x) / np.sum(self.g(u), axis=1)[..., np.newaxis]

    def j(self, u, d):
        return np.sum(self.g(u) * d)


class GustafsonKessel(ProbabilisticFuzzy):
    def distance(self, x, c, **kwargs):
        v = distance.Distance().vector_difference(x, c)
        e = self.e(x, c)
        pre = np.einsum('...ij,...jk', v, np.linalg.inv(e))
        return np.sum(pre * v, axis=2)

    def u(self, d):
        return super(GustafsonKessel, self).u(d)

    def c(self, x, u):
        return super(GustafsonKessel, self).c(x, u)

    def j(self, u, d):
        return super(GustafsonKessel, self).j(u, d)


class FCV(ProbabilisticFuzzy):
    def __init__(self, *args, **kwargs):
        super(FCV, self).__init__(*args, **kwargs)
        if 'r' in kwargs:
            self.r = kwargs['r']

    def distance(self, x, c, **kwargs):
        v = distance.vector_difference(x, c)
        b = self.b(x, c)
        return np.sum(v ** 2, axis=2) - np.sum(
            np.einsum('...ij,...jk', v, b), axis=2)

    def u(self, d):
        return super(FCV, self).u(d)

    def c(self, x, u):
        return super(FCV, self).c(x, u)

    def j(self, u, d):
        return super(FCV, self).j(u, d)

    def b(self, x, c):
        eigenvalues, eigenvectors = np.linalg.eig(self.e(x, c))
        idx = eigenvalues.argsort()[:, ::-1]
        sorted_eigenvectors = np.array(
            [eigenvectors[p][:, idx[p]] for p in range(len(c))])
        return sorted_eigenvectors[:, :, :self.r]


class RousseeuwTrauwaertKaufman(ProbabilisticFuzzy):
    def __init__(self, *args, **kwargs):
        super(RousseeuwTrauwaertKaufman, self).__init__(*args, **kwargs)
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']

    def g(self, u):
        return self.alpha * u + (1. - self.alpha) * u ** 2

    def u(self, d):
        beta = (1. - self.alpha) / (1. + self.alpha)
        if self.U is None:
            u_init = d ** -1. / np.sum(np.power(d, -1.), axis=0)
            return u_init
        else:
            u_old = self.U
            for i, (d, u) in enumerate(zip(d.T, u_old.T)):
                condition = u > 0
                q = np.sum(condition)
                u_new = ((1. + (q - 1.) * beta) / (
                    d * np.sum(np.divide(1., d[condition]))) - beta) / (
                            1. - beta)
                u_new[u_new < 0] = 0
                u_old[:, i] = u_new
            return u_old

    def c(self, x, u):
        return super(RousseeuwTrauwaertKaufman, self).c(x, u)

    def j(self, u, d):
        return super(RousseeuwTrauwaertKaufman, self).j(u, d)
