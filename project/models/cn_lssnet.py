"""Placeholder CN-LSSNet model definition."""

from project.models.base import BaseDetector


class CNLSSNetModel(BaseDetector):
    """Unified CN-LSSNet interface placeholder."""

    def fit(self, train_dataset, val_dataset=None, **kwargs):
        raise NotImplementedError("CNLSSNetModel.fit() is not implemented yet.")

    def predict_scores(self, dataset):
        raise NotImplementedError("CNLSSNetModel.predict_scores() is not implemented yet.")
