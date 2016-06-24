import numpy as np
import abc
from clustering.distance import Euclidean


class Validator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cluster_pattern):
        """Validate a given cluster pattern.

        Validators use information from a given clustering (in particular
        the partition) to produce a single index assessing the
        trustworthiness of that partition.

        Parameters
        ----------
        cluster_pattern : ClusterTools

        Returns
        -------

        """
        self.cluster_pattern = cluster_pattern
        self.data = cluster_pattern.learning_results.loadings
        self.partition = cluster_pattern.partition
        self.membership = cluster_pattern.learning_results.membership
        self.centers = cluster_pattern.learning_results.centers
        self.algorithm = cluster_pattern.learning_results.cluster_algorithm

    @abc.abstractmethod
    def validate(self):
        """Calculates the validator's index for its partition.

        Returns
        -------
        float
            The calculated index for this partition.

        """
        return

    @abc.abstractmethod
    def seeks(self, values):
        """The optimising condition of this validator.

        Parameters
        ----------
        values
            A list of validation indices.

        Returns
        -------
        function
            A function representing the optimum value of the validator,
            i.e. np.min or np.max

        """
        return

    def intercluster_distances(self):
        """Calculates the euclidean distance between cluster centers."""

        return Euclidean()(self.centers)


class DaviesBouldin(Validator):
    """Compares intercluster distance and cluster scatter.

    Specifically, two metrics are used: the distance between cluster centers,
    and the mean distance between the data and their respective clusters. The
    ratio between these is used to define the metric. Note that the
    Davies-Bouldin index requires hardened clusters.

    """

    def validate(self):
        m = self.intercluster_distances()
        s = self.cluster_scatter()
        r = (s + s[:, np.newaxis]) / m
        r[r == np.inf] = 0  # Same clusters shouldn't be compared.
        r = np.nan_to_num(r)  # nan means zero spread in this case.
        d = np.max(r, axis=0)
        return np.mean(d)

    @classmethod
    def seeks(cls, values):
        return np.argmin(values), np.min(values)

    def cluster_scatter(self):
        """Calculates the mean distance between data and cluster centers."""

        return np.array(
            [np.mean(Euclidean()(p, a.reshape(1, -1)), axis=1) for p, a in
             zip(self.partition, self.centers)]).flatten()


class Dunn(Validator):
    """Compares minimum intercluster distance with maximum cluster diameter.

    Dunn's index directly compares the smallest intercluster separation with
    the greatest intracluster diameter which can be measured in a number of
    ways. The validator here can be set to any of them.

    """

    def validate(self, intracluster_distance_metric='diameters'):
        if not hasattr(intracluster_distance_metric, '__call__'):
            try:
                idm = getattr(self,
                              intracluster_distance_metric.replace(' ', '_'))
            except AttributeError:
                raise AttributeError("'intracluster_distance metric' must be "
                                     "'diameters', 'mean pairwise distances', "
                                     "'mean distance from mean', "
                                     "or a function which returns a distance "
                                     "metric array for each partition.")
        else:
            idm = intracluster_distance_metric

        m = self.intercluster_distances()
        m[m == 0] = np.inf
        d = idm(self.partition)
        return np.min(m) / np.max(d)

    @classmethod
    def seeks(cls, values):
        return np.argmax(values), np.max(values)

    @staticmethod
    def diameters(partition):
        """Calculates the maximum pairwise distance across a partition.

        Parameters
        ----------
        partition : list
            a list of arrays, each containing data belonging to a specific
            cluster.

        """
        return np.array([np.max(Euclidean()(p)) for p in partition])

    @staticmethod
    def mean_pairwise_distances(partition):
        """Calculates the mean pairwise distance across a partition.

        Parameters
        ----------
        partition : list
            a list of arrays, each containing data belonging to a specific
            cluster.

        """
        mpd = [np.mean(Euclidean()(p)) / 2 for p in partition]
        return np.array(mpd)

    @staticmethod
    def mean_distance_from_mean(partition):
        """Calculates the data's mean distance from the partition's mean.

        Parameters
        ----------
        partition : list
            a list of arrays, each containing data belonging to a specific
            cluster.

        """
        ds = [np.mean(Euclidean()(p, np.mean(p, axis=0).reshape(1, -1))) for p
              in partition]
        return np.array(ds)


class PartitionCoefficient(Validator):
    """Specifically designed for fuzzy sets, the partition coefficient is based
    on the fuzzy membership matrix.
    """

    def validate(self):
        pc = np.trace(np.dot(self.membership.T,
                             self.membership) / self.membership.shape[1])
        return pc

    @classmethod
    def seeks(cls, values):
        return np.argmax(values), np.max(values)


class XieBeni(Validator):
    def validate(self):
        spread = np.square(Euclidean()(self.data, self.centers))
        separation = np.square(self.intercluster_distances())
        separation[separation == 0] = np.inf
        n = self.data.shape[0]
        return (1. / n) * np.sum(
            np.multiply(np.square(self.membership), spread)) / np.min(
            separation)

    @classmethod
    def seeks(cls, values):
        return np.argmin(values), np.min(values)


class PBMF(Validator):
    def validate(self):
        x = self.data
        u = self.membership
        v = self.centers
        k = u.shape[0]
        e1 = np.sum(np.square(u) * Euclidean()(x, np.mean(x, axis=0).reshape(
            1, -1)))
        ek = np.sum(np.square(u) * Euclidean()(x, v))
        dk = np.max(Euclidean()(v))
        return (1. / k * e1 / ek * dk) ** 2

    @classmethod
    def seeks(cls, values):
        return np.argmax(values), np.max(values)
