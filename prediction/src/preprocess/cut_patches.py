import numpy as np
import scipy.ndimage


def mm2voxel(coord, origin=0, spacing=1):
    coord = np.array(coord)
    origin = scipy.ndimage._ni_support._normalize_sequence(origin, len(coord))
    spacing = scipy.ndimage._ni_support._normalize_sequence(spacing, len(coord))
    coord = np.ceil(coord - np.array(origin)) / np.array(spacing)

    return coord.astype(np.int)


def cut_patch(ct_array, border_size, centroids, origin, spacing):
    border_size = scipy.ndimage._ni_support._normalize_sequence(border_size, len(ct_array.shape))
    padding = np.ceil(np.array(border_size) / 2.).astype(np.int)
    padding = np.stack([padding, padding], axis=1)
    ct_array = np.pad(ct_array, padding, mode='edge')
    for centroid in centroids:
        # TODO: create constant CT_DIM_ORDER = "xyz" | "zyx" or similar
        centroid = mm2voxel([centroid[i] for i in "xyz"], origin, spacing)

        yield ct_array[centroid[0]: centroid[0] + border_size[0],
                       centroid[1]: centroid[1] + border_size[1],
                       centroid[2]: centroid[2] + border_size[2]]


def patches_from_ct(ct_array, border_size, centroids, origin, spacing):
    patches = list()
    patch = cut_patch(ct_array, border_size, centroids, origin, spacing)
    while True:
        try:
            patches.append(next(patch))
        except StopIteration:

            return patches
