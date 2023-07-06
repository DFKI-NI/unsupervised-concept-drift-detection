from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier


class Classifiers:
    """
    Classifiers provides an interface to operate two HoeffdingTreeClassifiers and two GaussianNBs for a concept drift
    detector. One of each is assisted by the concept drift detector, whereas the remaining classifiers operate
    independently.
    """

    def __init__(self):
        """
        Init two HoeffdingTreeClassifiers and two GaussianNBs.
        """
        self.base_hoeffding_tree = HoeffdingTreeClassifier()
        self.base_gaussian_nb = GaussianNB()
        self.assisted_hoeffding_tree = HoeffdingTreeClassifier()
        self.assisted_gaussian_nb = GaussianNB()
        self.nonadaptive_trains = 0
        self.adaptive_trains = 0

    def predict(self, x):
        """
        Predict the label of the features x.

        :param x: the features
        :return: the label
        """
        predictions = (
            self.base_hoeffding_tree.predict_one(x),
            self.base_gaussian_nb.predict_one(x),
            self.assisted_hoeffding_tree.predict_one(x),
            self.assisted_gaussian_nb.predict_one(x),
        )
        return predictions

    def fit(self, x, y, nonadaptive):
        """
        Fit the classifiers on the training data consisting of x and y. If nonadaptive is True, the base classifiers are
        trained as well.

        :param x: the features
        :param y: the label
        :param nonadaptive: True if base classifiers shall be trained as well, else False
        """
        if nonadaptive:
            self.base_hoeffding_tree.learn_one(x, y)
            self.base_gaussian_nb.learn_one(x, y)
            self.nonadaptive_trains += 1
        self.adaptive_trains += 1
        self.assisted_hoeffding_tree.learn_one(x, y)
        self.assisted_gaussian_nb.learn_one(x, y)

    def reset(self):
        """
        Reset the classifiers assisted by concept drift detectors.
        """
        self.assisted_hoeffding_tree = HoeffdingTreeClassifier()
        self.assisted_gaussian_nb = GaussianNB()
