import pytest

from ..algorithms.classify import trained_model
from ..algorithms.classify.src.preprocess_patch import preprocess_LR3DCNN
from ..preprocess import preprocess_ct


@pytest.fixture
def dicom_path():
    yield '../images/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/' \
          '1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192'


@pytest.fixture
def model_path():
    yield '../classify_models/model.h5'


def test_classify_predict_model_load(dicom_path, model_path):
    predicted = trained_model.predict(dicom_path,
                                      [],
                                      model_path,
                                      preprocess_ct=None,
                                      preprocess_model_input=preprocess_LR3DCNN)

    assert len(predicted) == 0


def test_classify_predict_inference(dicom_path, model_path):
    params = preprocess_ct.Params(clip_lower=-1000,
                                  clip_upper=400,
                                  spacing=(.6, .6, .3))
    preprocess = preprocess_ct.PreprocessCT(params)
    predicted = trained_model.predict(dicom_path,
                                      [{'x': 50, 'y': 50, 'z': 21}],
                                      model_path,
                                      preprocess_ct=preprocess,
                                      preprocess_model_input=preprocess_LR3DCNN)

    assert len(predicted) == 1
    assert isinstance(predicted[0]['p_concerning'], float)
    assert predicted[0]['p_concerning'] >= 0.
    assert predicted[0]['p_concerning'] <= 1.
