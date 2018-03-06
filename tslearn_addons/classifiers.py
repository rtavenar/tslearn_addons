import numpy as np

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import to_time_series_dataset
from tslearn.barycenters import EuclideanBarycenter, DTWBarycenterAveraging, SoftDTWBarycenter

class SupervisedTimeSeriesKMeans(TimeSeriesKMeans):
    """
    Supervised K-means clustering for timeseries data.
    This classifier uses the same method for classification as TimeSeriesKMeans: 
    The timeseries' class is the cluster, thats barycenter has the smallest distance 
    to the timeseries. The only difference lies within training:
    If the labels are known, the clusters themselves don't have to be determined,
    they are given immediately via the labels. So, for training it's enough to  
    calculate for all classes the barycenter of the timeseries within that class.

    Parameters
    ----------
    metric : {"euclidean", "dtw", "softdtw"} (default: "euclidean")
        Metric to be used for both cluster assignment and barycenter computation. 
        If "dtw", DBA is used for barycenter computation.
    metric_params : dict or None
        Parameter values for the chosen metric. Value associated to the `"gamma_sdtw"` 
        key corresponds to the gamma parameter in Soft-DTW.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point.
    cluster_centers_ : numpy.ndarray
        Cluster centers.

    Note
    ----
        If `metric` is set to `"euclidean"`, the algorithm expects a dataset of 
        equal-sized time series.
    """
    
    def __init__(self, metric="euclidean", metric_params=None):
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y):  
        """Compute supervised k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        y : array-like of shape=(n_ts,)
            Time series labels to fit.
        """

        cls, self.labels_ = np.unique(y, return_inverse=True)
        self.n_clusters = len(cls)
        if self.metric_params is None:
            self.metric_params = {}
        self.gamma_sdtw = self.metric_params.get("gamma_sdtw", 1.)

        self.Xs_ = []
        self.ys_ = []
        centroids = []
        for i in range(self.n_clusters):
            self.Xs_.append(to_time_series_dataset(X[self.labels_==i,:,:]))
            self.ys_.append(self.labels_[self.labels_==i])

            if self.metric == 'euclidean':
                centroids.append(EuclideanBarycenter().fit(self.Xs_[i]))
            if self.metric == 'dtw':
                centroids.append(DTWBarycenterAveraging().fit(self.Xs_[i]))
            if self.metric == 'softdtw':
                centroids.append(SoftDTWBarycenter().fit(self.Xs_[i]))

        self.cluster_centers_ = np.stack([centroids]).squeeze()
        return self

    def fit_predict(self, X, y):
        """Fit supervised k-means clustering using X and y and then predict the 
        closest cluster each time series in X belongs to.
        This overrides TimeSeriesKMeans.fit_predict(X, y=None), s.t. it raises 
        whenever no labels are passed.
        It is more efficient to use this method than to sequentially call fit and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.
        y : array-like of shape=(n_ts,)
            Time series labels to fit.

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_
