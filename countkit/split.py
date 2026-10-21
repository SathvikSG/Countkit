"""Split touching objects with a distance-transform watershed."""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def split_touching(mask, min_distance=5, smooth=0.0):
    """Label a binary mask, cutting apart blobs that touch.

    Raise min_distance or smooth if single objects get cut in pieces;
    lower them if touching objects stay merged.
    """
    distance = ndi.distance_transform_edt(mask)
    if smooth:
        distance = ndi.gaussian_filter(distance, smooth)
    coarse, _ = ndi.label(mask)
    peaks = peak_local_max(
        distance, min_distance=min_distance, labels=coarse, exclude_border=False
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    # Blobs too small to have a peak still need a marker, or they vanish.
    missing = set(np.unique(coarse)) - {0} - set(np.unique(coarse[markers > 0]))
    next_id = len(peaks) + 1
    for lab in missing:
        ys, xs = np.nonzero(coarse == lab)
        i = len(ys) // 2
        markers[ys[i], xs[i]] = next_id
        next_id += 1
    return watershed(-distance, markers, mask=mask)
