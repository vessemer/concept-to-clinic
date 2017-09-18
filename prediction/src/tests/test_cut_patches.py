import numpy as np
import pytest

from ..preprocess import cut_patches
from ..preprocess import load_ct


@pytest.fixture(scope='session')
def get_patches(tmpdir_factory):
    ct_array, meta = load_ct.load_ct('../images/LUNA-0001/'
                                     + '1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797')

    centroids = [[0, 0, 0], [0, 12, 24], [45, 45, 12]]
    centroids = [{'x': centroid[0], 'y': centroid[1], 'z': centroid[2]} for centroid in centroids]
    patches = cut_patches.patches_from_ct(ct_array, border_size=12, centroids=centroids, origin=0, spacing=1)
    assert isinstance(patches, list)
    assert len(patches) == 3
    assert all([patch.shape == (12, 12, 12) for patch in patches])

    path = str(tmpdir_factory.mktemp("patches").join("patches.npy"))
    np.save(path, np.asarray(patches))
    return path


def test_tmp(get_patches):
    patches = np.load(get_patches)
    #   Despite they are overlapped, the amount of volumes must have preserved
    assert patches.shape == (3, 12, 12, 12)

#
#
# def test_calculate_volume_over_connected_components_with_dicom_path(get_mask_connected):
#     path, centroids = get_mask_connected
#     voxels_volumes = [100, 100, 30]
#     dicom_path = '../images/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/' \
#                  '1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192'
#     real_volumes = trained_model.calculate_volume(str(path), centroids, dicom_path)
#
#     #   Despite they are overlapped, the amount of volumes must have preserved
#     assert len(real_volumes) == len(voxels_volumes)
#     assert all([1.2360 >= mm / vox >= 1.2358
#                 for vox, mm in zip(voxels_volumes, real_volumes)])
