"""
The :mod:`tslearn_addons.metrics` module gathers time series similarity metrics.
"""

import numpy
from tslearn.cylrdtw import lr_dtw as cylr_dtw, lr_dtw_backtrace as cylr_dtw_backtrace, cdist_lr_dtw as cycdist_lr_dtw
from tslearn.utils import to_time_series, to_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def lr_dtw(s1, s2, gamma=0.):
    """Compute Locally Regularized DTW (LR-DTW) similarity measure between (possibly multidimensional) time series and
    return it.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1
        A time series
    s2
        Another time series
    gamma : float (default: 0.)
        Regularization parameter

    Returns
    -------
    float
        Similarity score

    See Also
    --------
    lr_dtw_path : Get both the matching path and the similarity score for LR-DTW
    cdist_lr_dtw : Cross similarity matrix between time series datasets
    dtw : Dynamic Time Warping score
    dtw_path : Get both the matching path and the similarity score for DTW
    """
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    return cylr_dtw(s1, s2, gamma=gamma)[0]


def cdist_lr_dtw(dataset1, dataset2=None, gamma=0.):
    """Compute cross-similarity matrix using Locally-Regularized Dynamic Time Warping (LR-DTW) similarity measure.

    Parameters
    ----------
    dataset1 : array-like
        A dataset of time series
    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of `dataset1` is returned.
    gamma : float (default: 0.)
        :math:`\\gamma` parameter for the LR-DTW metric.

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    See Also
    --------
    lr_dtw : Get LR-DTW similarity score
    """
    dataset1 = to_time_series_dataset(dataset1)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = to_time_series_dataset(dataset2)
    return cycdist_lr_dtw(dataset1, dataset2, gamma=gamma, self_similarity=self_similarity)


def lr_dtw_path(s1, s2, gamma=0.):
    """Compute Locally Regularized DTW (LR-DTW) similarity measure between (possibly multidimensional) time series and
    return both the (probabilistic) path and the similarity.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1
        A time series
    s2
        Another time series
    gamma : float (default: 0.)
        Regularization parameter

    Returns
    -------
    numpy.ndarray of shape (s1.shape[0], s2.shape[0])
        Matching path represented as a probability map
    float
        Similarity score

    See Also
    --------
    lr_dtw : LR-DTW score
    dtw : Dynamic Time Warping (DTW) score
    dtw_path : Get both the matching path and the similarity score for DTW
    """
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    sim, probas = cylr_dtw(s1, s2, gamma=gamma)
    path = cylr_dtw_backtrace(probas)
    return path, sim
