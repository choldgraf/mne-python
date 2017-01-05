"""
=========================================
Receptive Field Estimation and Prediction
=========================================

This example reproduces figures from Lalor et al's mTRF toolbox in
matlab [1]_. We will show how the :class:`mne.decoding.ReceptiveField` class
can perform a similar function along with :mod:`sklearn`. We will fit a
linear encoding model using the continuously-varying speech envelope to
predict activity of a 128 channel EEG system.

References
----------
.. [1] Crosse, M. J., Di Liberto, G. M., Bednar, A. & Lalor, E. C. (2016).
       The Multivariate Temporal Response Function (mTRF) Toolbox:
       A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli.
       Frontiers in Human Neuroscience 10, 604. doi:10.3389/fnhum.2016.00604

.. _figure 1: http://journal.frontiersin.org/article/10.3389/fnhum.2016.00604/full#F1
.. _figure 2: http://journal.frontiersin.org/article/10.3389/fnhum.2016.00604/full#F2
"""  # noqa: E501

# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import pearsonr

import mne
from mne.decoding import ReceptiveField
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale


###############################################################################
# Load the data from the publication
# ----------------------------------
#
# First we will load the data collected in [1]_. In this experiment subjects
# listened to natural speech. Raw EEG and the speech stimulus was collected.
# We will load these below, downsampling the data in order to speed up
# computation since we know that our features are primarily low-frequency in
# nature. Then we'll visualize both the EEG and speech envelope.

path = mne.datasets.mtrf.data_path()
n_decim = 2
data = loadmat(path + '/speech_data.mat')
raw = data['EEG'].T
speech = data['envelope'].T
sfreq = float(data['Fs'])
sfreq /= n_decim
speech = mne.filter.resample(speech, down=n_decim, npad='auto')
raw = mne.filter.resample(raw, down=n_decim, npad='auto')


# Read in channel positions and create our MNE objects from the raw data
montage = mne.channels.read_montage('biosemi128')
montage.selection = montage.selection[:128]
info = mne.create_info(montage.ch_names[:128], sfreq, 'eeg', montage=montage)
raw = mne.io.RawArray(raw, info)
n_channels = len(raw.ch_names)

# Plot a sample of brain and stimulus activity
fig, ax = plt.subplots()
ax.plot(scale(raw[:][0][..., :800].T), color='k', alpha=.1)
ax.plot(scale(speech[0, :800]), color='r', lw=2)
ax.set(title="Sample brain and stimulus activity", xlabel="Time (s)")
mne.viz.tight_layout()

###############################################################################
# Create and fit a receptive field model
# --------------------------------------
#
# We will construct a model to find the linear relationship between the EEG
# signal and a time-delayed version of the speech envelope. This allows
# us to make predictions about the response to new stimuli.

# Define the delays that we will use in the receptive field
tmin, tmax = -.4, .2

# Initialize the model
spec_names = ['freq_%s' % ii for ii in range(speech.shape[0])]
rf = ReceptiveField(tmin, tmax, sfreq,
                    feature_names=spec_names, model=Ridge(alpha=1.))
n_delays = int((tmax - tmin) * sfreq) + 2  # +2 to account for 0 + end

n_splits = 3
cv = KFold(n_splits)

# Prepare model data
Y, _ = raw[:]  # Outputs for the model
Y = np.rollaxis(Y, -1, 0)  # Make time the first dimension
speech = np.rollaxis(speech, -1, 0)

# Iterate through folds, fit the model, and predict/test on held-out data
coefs = np.zeros([n_splits, n_channels, n_delays])
scores = np.zeros([n_splits, n_channels])
for ii, (train, test) in enumerate(cv.split(speech)):
    print('CV iteration %s' % ii)
    rf.fit(speech[train], Y[train])
    preds = rf.predict(speech[test])

    # mask_pred tells us which points were removed / kept in time delaying
    for jj, i_ch in enumerate(Y.T):
        scores[ii, jj] = pearsonr(Y[:, jj][test][rf.mask_predict_],
                                  preds[:, jj])[0]
    # Remove features dimension because there's only one item in it
    coefs[ii] = rf.coef_
delays_time = rf.delays_ / float(sfreq)
# Average scores and coefficients across CV splits
mn_coefs = coefs.mean(0)
mn_scores = scores.mean(0)

# Plot mean prediction scores across all channels
fig, ax = plt.subplots()
ix_chs = np.arange(n_channels)
ax.plot(ix_chs, mn_scores)
ax.axhline(0, ls='--', color='r')
ax.set(title="Mean prediction score", xlabel="Channel", ylabel="Score ($r$)")
mne.viz.tight_layout()

###############################################################################
# Investigate model coefficients
# ------------------------------
#
# Finally, we will look at how the linear coefficients (sometimes
# referred to as beta values) are distributed across time delays as well as
# across the scalp. We will recreate `figure 1`_ and `figure 2`_ from [1]_.

# Print mean coefficients across all time delays / channels (see Fig 1 in [1])
time_plot = -.180  # For highlighting a specific time.
fig, ax = plt.subplots(figsize=(4, 8))
vmax = np.abs(mn_coefs.max())
vmin = -vmax
ax.pcolormesh(delays_time, ix_chs, mn_coefs, cmap='RdBu_r',
              vmin=vmin, vmax=vmax)
ax.axvline(time_plot, ls='--', color='k', lw=2)
ax.set(xlabel='Lag (s)', ylabel='Channel', title="Mean Model\nCoefficients",
       xlim=[delays_time.min(), delays_time.max()], ylim=[len(ix_chs) - 1, 0])
plt.setp(ax.get_xticklabels(), rotation=45)
mne.viz.tight_layout()

# Make a topographic map of coefficients for a given delay (see Fig 2C in [1])
ix_plot = np.argmin(np.abs(time_plot - delays_time))
fig, ax = plt.subplots()
mne.viz.plot_topomap(mn_coefs[:, ix_plot], pos=info, axes=ax, show=False,
                     vmin=vmin, vmax=vmax)
ax.set(title="Topomap of model coefficicients\nfor delay %s" % time_plot)
mne.viz.tight_layout()

plt.show()
