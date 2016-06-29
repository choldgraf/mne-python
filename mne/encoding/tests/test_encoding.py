# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Jean-Remi King <jeanremi.king@gmail.com>
#          Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
import warnings
import os.path as op

from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne import io, Epochs, read_events, pick_types
from mne.utils import (requires_sklearn, run_tests_if_main)


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

np.random.seed(1337)

tmin, tmax = -0.1, 0.5
event_id = dict(aud_l=1, vis_l=3)

warnings.simplefilter('always')

# Loading raw data + epochs
raw = io.read_raw_fif(raw_fname, preload=True)
events = read_events(event_name)
picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                   eog=False, exclude='bads')
picks = picks[0:2]

with warnings.catch_warnings(record=True):
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, preload=True)


@requires_sklearn
def test_rerp():
    from mne.encoding import EventRelatedRegressor, clean_inputs
    rerp = EventRelatedRegressor(raw, events, est='cholesky',
                                 event_id=event_id, tmin=tmin, tmax=tmax,
                                 preproc_func_xy=clean_inputs, picks=picks)
    rerp.fit()

    cond = 'aud_l'
    evoked_erp = rerp.to_evoked()[cond]
    evoked_avg = epochs[cond].average()

    assert_array_almost_equal(evoked_erp.data, evoked_avg.data, 12)

    # Make sure events are MNE-style
    raw_nopreload = io.read_raw_fif(raw_fname, preload=False)
    assert_raises(ValueError, EventRelatedRegressor, raw, events[:, 0])
    # Data needs to be preloaded
    assert_raises(ValueError, EventRelatedRegressor, raw_nopreload, events)
    # Data must be fit before we get evoked coefficients
    rerp = EventRelatedRegressor(raw, events, est='cholesky',
                                 event_id=event_id, tmin=tmin, tmax=tmax,
                                 preproc_func_xy=clean_inputs, picks=picks)
    assert_raises(ValueError, rerp.to_evoked)


@requires_sklearn
def test_custom():
    from mne.encoding import DataDelayer
    from mne.encoding.model import EncodingModel
    # Simple regression
    a = np.random.randn(10000)
    w = .2
    b = a * w
    enc = EncodingModel()
    enc.fit(a[:, np.newaxis], b)
    assert_array_almost_equal(enc.est._final_estimator.coef_, w)

    # Now w/ delays
    sfreq = 100
    w = [.2, .7, -.4]
    delayer = DataDelayer(delays=[-.1, 0, .1], sfreq=sfreq)
    a_del = delayer.fit_transform(a[:, np.newaxis])
    b_del = np.dot(a_del, w)

    enc.fit(a_del, b_del)
    assert_array_almost_equal(enc.est._final_estimator.coef_[0], w)

    # Y must be given
    assert_raises(ValueError, enc.fit, a_del, None)
    # X/y must have same first dim
    assert_raises(ValueError, enc.fit, a_del[:-1], b_del)
    # Wrong est
    assert_raises(ValueError, EncodingModel, est='foo')
    assert_raises(ValueError, EncodingModel, est=delayer)


@requires_sklearn
def test_feature():
    from mne.encoding import (DataSubsetter, EventsBinarizer, DataDelayer,
                              clean_inputs)
    sfreq = raw.info['sfreq']
    # Delayer must have sfreq if twin given
    assert_raises(ValueError, DataDelayer, time_window=[tmin, tmax],
                  sfreq=None)
    # Must give either twin or delays
    assert_raises(ValueError, DataDelayer, time_window=[tmin, tmax],
                  sfreq=sfreq, delays=[1.])
    assert_raises(ValueError, DataDelayer)
    # EventsBinarizer must have proper events shape
    binarizer = EventsBinarizer(raw.n_times)
    assert_raises(ValueError, binarizer.fit, events)
    # Subsetter works for indexing
    data = np.arange(100)[:, np.newaxis]
    sub = DataSubsetter(np.arange(50))
    data_subset = sub.fit_transform(data)
    assert_array_equal(data_subset, data[:50])
    # Subsetter works for decimation
    sub = DataSubsetter(decimate=10)
    data_subset = sub.fit_transform(data)
    assert_array_equal(data_subset, data[::10])

    # Subsetter indices must not exceed length of data
    sub = DataSubsetter([1, 99999999])
    assert_raises(ValueError, sub.fit, raw._data.T)
    # Cleaning inputs must have same n times
    assert_raises(ValueError, clean_inputs, raw._data.T, raw._data[0, :-1].T)


run_tests_if_main()
