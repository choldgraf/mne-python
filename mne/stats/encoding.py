from __future__ import division
import numpy as np
from scipy.linalg import svd
from sklearn.cross_validation import KFold, LabelShuffleSplit, LeavePLabelOut
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from mne.utils import _time_mask
from copy import deepcopy

__all__ = ['EncodingModel',
           'delay_timeseries']


class EncodingModel(object):
    def __init__(self, delays=None, est=None,
                 cv=None, scorer=None):
        """Fit a STRF model.

        This implementation uses Ridge regression and scikit-learn. It creates time
        lags for the input matrix, then does cross validation to fit a STRF model.

        Parameters
        ----------
        X : array, shape (n_epochs, n_feats, n_times)
            The input data for the regression
        y : array, shape (n_times,)
            The output data for the regression
        sfreq : float
            The sampling frequency for the time dimension
        delays : array, shape (n_delays,)
            The delays to include when creating time lags. The input array X will
            end up having shape (n_feats * n_delays, n_times)
        tmin : float | array, shape (n_epochs,)
            The beginning time for each epoch. Optionally a different time for each
            epoch may be provided.
        tmax : float | array, shape (n_epochs,)
            The end time for each epoch. Optionally a different time for each
            epoch may be provided.
        est : list (instance of sklearn, dict of params)
            A list specifying the model and parameters to use. First item must be a
            sklearn regression-style estimator. Second item is a
            dictionary of kwargs to pass in the construction of that estimator. If
            any values in kwargs is len > 1, then it is assumed that an inner CV
            loop is required to select the best value using GridSearchCV.
        cv_outer : int | instance of (KFold, LabelShuffleSplit)
            The cross validation object to use for the outer loop
        cv_inner : int | instance of same type as cv_outer
            The cross validation object to use for the inner loop,
            if hyperparameters are to be chosen computationally.
        scorer_outer : function | None
            The scorer to use when evaluating on the held-out test set.
            It must accept two 1-d arrays as inputs, and output a scalar value.
            If None, it will be correlation.
        X_names : list of strings/ints/floats, shape (n_feats,) : None
            A list of values corresponding to input features. Useful for keeping
            track of the coefficients in the model after time lagging.
        scale_data : bool
            Whether or not to scale the data to 0 mean and unit var before fit.

        Outputs
        -------
        ests : list of sklearn estimators, length (len(cv_outer),)
            The estimator fit on each cv_outer loop. If len(hyperparameters) > 1,
            then this will be the chosen model using GridSearch on each loop.
        scores: array, shape (len(cv_outer),)
            The scores on the held out test set on each loop of cv_outer
        X_names : array of strings, shape (n_feats * n_delays)
            A list of names for each coefficient in the model. It is of structure
            'name_timedelay'.
        """
        self.delays = np.array([0]) if delays is None else delays
        self.n_delays = len(self.delays)
        self.est = Ridge() if est is None else est
        if isinstance(cv, (float, int)):
            self.cv = KFold(n_folds=cv)
        else:
            self.cv = cv
        self.scorer = mean_squared_error if scorer is None else scorer

    def fit(self, X, y, sfreq, times=None, tmin=None, tmax=None, feat_names=None):
        """Fit the model"""
        if feat_names is not None:
            if len(feat_names) != X.shape[1]:
                raise ValueError('feat_names and X.shape[0] must be the same size')
        if times is None:
            times = np.arange(X.shape[-1]) / float(sfreq)
        self.tmin = times[0] if tmin is None else tmin
        self.tmax = times[-1] if tmax is None else tmax
        self.times = times
        self.sfreq = sfreq
        self.feat_names = [str(i) for i in range(len(X))] if feat_names is None else feat_names
        self.feat_names = np.array(self.feat_names)

        # Delay X
        X, y, labels = _build_design_matrix(X, y, sfreq, self.times,
                                            self.delays, self.tmin, self.tmax)
        cv = _check_cv(X, labels, self.cv)

        # Define names for input variabels to keep track of time delays
        X_names = ['{0}_{1}'.format(feat, delay)
                   for delay in self.delays for feat in self.feat_names]
        self.feature_names_with_lags_ = X_names

        # Build model instance
        if not isinstance(self.est, Pipeline):
            self.est = Pipeline([('est', self.est)])

        # Fit the models
        if isinstance(self.est.steps[-1][-1], GridSearchCV):
            # Assume hyperparameter search
            self.est.fit(X, y)
            grid = self.est.steps[-1][-1]
            if grid.refit:
                self.best_estimator_ = grid.best_estimator_
            self.best_params_ = grid.best_params_
            self.grid_scores_ = grid.grid_scores_
        else:
            # Assume regular model fit + keeping models
            coefs, scores = [[] for _ in range(2)]
            for i, (tr, tt) in enumerate(cv):
                X_tr = X[:, tr].T
                X_tt = X[:, tt].T
                y_tr = y[tr]
                y_tt = y[tt]
                lab_tr = labels[tr]
                lab_tt = labels[tt]

                # Fit model + make predictions
                self.est.fit(X_tr, y_tr)
                scr = self.scorer(self.est.predict(X_tt), y_tt)

                scores.append(scr)
                coefs.append(self.est.steps[-1][-1].coef_)
            self.coefs_all_ = np.array(coefs)
            self.coefs_ = np.mean(self.coefs_all_, axis=0)
            self.scores_ = np.array(scores)

    def predict(self, X):
        X_lag = delay_timeseries(X, self.sfreq, self.delays)

        Xt = self.est._pre_transform(X_lag.T)[0]
        return np.dot(Xt, self.coefs_)

    def plot_coefficients(self, agg=np.mean, ax=None, **kwargs):
        from matplotlib import pyplot as plt
        coefs = agg(self.coefs_all_, axis=0)
        coefs = coefs.reshape([-1, self.n_delays])

        if ax is None:
            f, ax = plt.subplots()
        im = ax.imshow(self.coefs_.reshape([-1, self.n_delays]),
                       **kwargs)

        for lab in ax.get_xticklabels():
            lab.set_text(self.delays[int(lab.get_position()[0])])

        for lab in ax.get_yticklabels():
            lab.set_text(self.feat_names[int(lab.get_position()[1])])

        ax.set_xlabel('Time delays (s)')
        ax.set_ylabel('Features')
        return ax


def delay_timeseries(ts, sfreq, delays):
    """Include time-lags for a timeseries.

    Parameters
    ----------
    ts: array, shape(n_feats, n_times)
        The timeseries to delay
    sfreq: int
        The sampling frequency of the series
    delays: list of floats
        The time (in seconds) of each delay
    Returns
    -------
    delayed: array, shape(n_feats*n_delays, n_times)
        The delayed matrix
    """
    delayed = []
    for delay in delays:
        roll_amount = int(delay * sfreq)
        rolled = np.roll(ts, roll_amount, axis=1)
        if delay < 0:
            rolled[:, roll_amount:0] = 0
        elif delay > 0:
            rolled[:, 0:roll_amount] = 0
        delayed.append(rolled)
    delayed = np.vstack(delayed)
    return delayed


def _scorer_corr(x, y):
    return np.corrcoef(x, y)[1, 0]


def _check_time(X, time):
    if isinstance(time, (int, float)):
        time = np.repeat(time, X.shape[0])
    elif time.shape[0] != X.shape[0]:
        raise ValueError('time lims and X must have the same shape')
    return time


def _check_inputs(X, y, times, delays, tmin, tmax):
    # Add an epochs dimension
    if X.ndim == 2:
        X = X[np.newaxis, ...]
    if y.ndim == 1:
        y = y[np.newaxis, ...]

    if not X.shape[-1] == y.shape[-1] == times.shape[-1]:
        raise ValueError('X, y, or times have different time dimension')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y have different number of epochs')
    if any([tmin + np.min(delays) < np.min(times),
            tmax + np.max(delays) > np.max(times)]):
        raise ValueError('Data will be cut off w delays, use longer epochs')
    tmin = _check_time(X, tmin)
    tmax = _check_time(X, tmax)
    return X, y, tmin, tmax


def _build_design_matrix(X, y, sfreq, times, delays, tmin, tmax):
    X, y, tmin, tmax = _check_inputs(X, y, times, delays, tmin, tmax)
    # Iterate through epochs with custom tmin/tmax if necessary
    X_out, y_out, lab_out = [[] for _ in range(3)]
    for i, (epX, epy, tmin, tmax) in enumerate(zip(X, y, tmin, tmax)):
        msk_time = _time_mask(times, tmin, tmax)

        epX_del = delay_timeseries(epX, sfreq, delays)
        epX_out = epX_del[:, msk_time]
        epy_out = epy[msk_time]
        ep_lab = np.repeat(i + 1, epy_out.shape[-1])

        X_out.append(epX_out)
        y_out.append(epy_out)
        lab_out.append(ep_lab)
    return np.hstack(X_out), np.hstack(y_out), np.hstack(lab_out)


def _check_cv(X, labels, cv):
    cv = 5 if cv is None else cv
    if isinstance(cv, float):
        raise ValueError('cv must be an int or instance of sklearn cv')
    if isinstance(cv, int):
        if len(np.unique(labels)) == 1:
            # Assume single continuous data, do KFold
            cv = KFold(labels.shape[-1], cv)
        else:
            # Assume trials structure, do LabelShufleSplit
            cv = LabelShuffleSplit(labels, n_iter=cv, test_size=.2)
    return cv
