# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
import warnings
import os.path as op

from nose.tools import assert_raises, assert_true, assert_equal
import numpy as np
from numpy.testing import assert_array_equal

from mne import io, read_events, pick_types
from mne.utils import (requires_sklearn, run_tests_if_main)


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

np.random.seed(1337)

tmin, tmax = -0.1, 0.5
event_id = dict(aud_l=1, vis_l=3)

warnings.simplefilter('always')

# Loading raw data
raw = io.read_raw_fif(raw_fname, preload=True)
picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                   eog=False, exclude='bads')
picks = picks[0:2]


@requires_sklearn
def test_feature():
    from mne.encoding import (SampleMasker, EventsBinarizer,
                              FeatureDelayer)
    from sklearn.linear_model import Ridge
    from scipy.sparse import csr_matrix
    events = read_events(event_name)
    events[:, 0] -= raw.first_samp

    # --- Feature Delayer ---
    # Explicit delays + sfreq
    X = np.random.randn(1000, 2)
    for idel, isfreq in [[(0, 1, 2), 1], [(0, .1, .2), 10]]:
        delayer = FeatureDelayer(delays=idel, sfreq=isfreq)
        Xdel = delayer.transform(X)
        assert_array_equal(Xdel[:, 0], X[:, 0])
        assert_array_equal(Xdel[:-1, 1], X[1:, 0])

    # Sparse matrices
    X = np.zeros([100, 2])
    X[10, 0] = 1
    X = csr_matrix(X)
    Xdel = delayer.transform(X)
    assert_true(isinstance(Xdel, csr_matrix))
    assert_equal(Xdel[10, 0], X[10, 0])

    # --- Events Binarizer ---
    # EventsBinarizer must have proper events shape
    binarizer = EventsBinarizer(raw.n_times)
    assert_raises(ValueError, binarizer.transform, events)

    # Test outputs are correct when working properly
    events = events[events[:, 0] <= raw.n_times, :]
    binarizer = EventsBinarizer(raw.n_times, sparse=True)
    ev_cont = binarizer.transform(events[:, 0], events[:, 2])

    # Covariates
    cov = np.random.randn(events.shape[0], 2)
    ev_cont = binarizer.transform(events[:, 0], events[:, 2],
                                  covariates=cov)
    assert_equal(ev_cont.shape[-1],
                 len(set(events[:, 2])) + cov.shape[-1])
    new_names = (['event_%s' % e for e in set(events[:, 2])] +
                 ['cov_0', 'cov_1'])
    covs_new = ev_cont[:, -2:].toarray()
    covs_new = covs_new[(covs_new != 0).all(1), :]
    assert_array_equal(cov, covs_new)
    assert_array_equal(set(new_names), set(binarizer.names_))
    # Make sure covariates are correct shape / num
    assert_raises(ValueError, binarizer.transform, events[:, 0],
                  covariates=cov[:-1])
    assert_raises(ValueError, binarizer.transform, events[:, 0],
                  covariates=cov[np.newaxis, ...])
    # Cov names
    assert_raises(ValueError, binarizer.transform, events,
                  covariates=cov, covariate_names=['foo', 'bar', 'haz'])
    ev_cont = binarizer.transform(events[:, 0], events[:, 2],
                                  covariates=cov,
                                  covariate_names=['foo', 'bar'])
    assert_equal(['foo', 'bar'], binarizer.names_[-2:])

    # --- SampleMasker ---
    # Subsetter works for indexing
    data = np.arange(100)[:, np.newaxis]
    masker = SampleMasker(Ridge(), ixs=data[:50, 0], ixs_pred=data[50:, 0])
    assert_array_equal(data[masker.ixs], data[:50])
    assert_array_equal(data[masker.ixs_pred], data[50:])
    # Subsetter indices must not exceed length of data
    sub = SampleMasker(Ridge(), ixs=[1, 99999999])
    assert_raises(ValueError, sub.fit, data, data[:, 0])
    # Create data
    X = np.tile(np.arange(100), [10, 1]).T.astype(float)
    y = np.arange(100)
    mod = SampleMasker(Ridge(), mask_val=np.nan)

    # This should remove no datapoints
    mod.fit(X, y)
    assert_true(mod.mask.sum() == X.shape[0])

    # Test that it removes nans
    X[:20, :] = np.nan
    mod.fit(X, y)
    assert_true(mod.mask.sum() == (X.shape[0] - 20))
    # Make sure the right indices were removed
    assert_true((np.isnan(X[~mod.mask]).all().all()))

    # Ensure that other numbers work
    X = np.tile(np.arange(100), [10, 1]).T.astype(float)
    y = np.arange(100)
    mod = SampleMasker(Ridge(), mask_val=10)
    mod.fit(X, y)
    assert_true(np.where(~mod.mask)[0][0] == mod.mask_val)

    # Make sure a callable returns a 1-d output
    mod = SampleMasker(Ridge(), mask_val=lambda a: a < 5)
    assert_raises(ValueError, mod.fit, X, y)
    # Make sure callable works
    mod = SampleMasker(Ridge(), mask_val=lambda a: (a < 5).all(1))
    mod.fit(X, y)
    assert_equal(mod.mask.sum(), 95)

    # Prediction ixs
    mod = SampleMasker(Ridge(), mask_val=np.nan, ixs_pred=[1, 2, 3])
    mod.fit(X, y)
    ypred = mod.predict(X)
    assert_equal(ypred.shape[0], 3)

    # Misc
    assert_raises(SampleMasker, mask_val=0, ixs=[1, 2])
    assert_raises(SampleMasker, mask_val=0, mask_condition='foo')


@requires_sklearn
def test_encoding():
    from mne.encoding import (SampleMasker, get_coefs, get_final_est)
    from mne.encoding.model import _check_estimator
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.linear_model import Ridge
    # Make sure estimator pulling works
    mod = Ridge()
    pipe = make_pipeline(Ridge())
    samp = SampleMasker(pipe)
    est = get_final_est(pipe)
    assert_true(isinstance(est, type(mod)))
    est = get_final_est(samp)
    assert_true(isinstance(est, type(mod)))
    # Est must be fit first
    assert_raises(ValueError, get_coefs, est)
    # Coefs are correctly taken
    est.fit([[1, 2], [3, 4]], [1, 2])
    coefs = get_coefs(get_final_est(samp), 'coef_')
    assert_equal(coefs.shape[-1], 2)
    # Incorrect coefficient name
    assert_raises(ValueError, get_coefs, est, 'foo')

    # Make sure the checks are working
    # None returns Ridge instance
    assert_true(isinstance(_check_estimator(None), Pipeline))
    assert_true(isinstance(get_final_est(_check_estimator(None)), type(mod)))
    # Correct ridge solver
    assert_equal(get_final_est(_check_estimator('lsqr')).solver, 'lsqr')
    # Incorrect string type
    assert_raises(ValueError, _check_estimator, 'foo')
    # Strings return an estimator instance
    assert_true(isinstance(get_final_est(_check_estimator(None)), Ridge))
    # Estimator must have fit/predict methods
    assert_raises(ValueError, _check_estimator, lambda a: a + 1)

run_tests_if_main()
