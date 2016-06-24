import random

import nose.tools as nt
from sklearn import datasets

from clustering import initialization


class TestInitialization:

    def setUp(self):
        self.n_variables = random.randint(2, 10)
        self.n_centers = 2
        self.data = datasets.make_blobs(centers=self.n_centers, n_features=self.n_variables)[0]

    def test_shape(self):
        methods = [initialization.statistical_guess]
        for method in methods:
            yield self.check_shape, method

    def check_shape(self, method):
        cluster_centers = method(self.data, self.n_centers)
        nt.assert_equal(cluster_centers.shape, (self.n_centers, self.n_variables))
