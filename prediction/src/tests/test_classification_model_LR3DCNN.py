import pytest

from prediction.src.algorithms.classify.src.lr3dcnn import model


@pytest.fixture
def metaimage_path():
    yield '../images/LUNA-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797'


@pytest.fixture
def model_path():
    yield '../classify_models/'


def test_classification_model_lr3dcnn_compile():
    classify = model.Model(init_model=False)
    classify.init_model()


def test_classification_model_lr3dcnn_load(model_path):
    classify = model.Model(init_model=False)
    classify.init_model()
    classify.load_model(model_path)


def test_classification_model_lr3dcnn_train(model_path):
    classify = model.Model(init_model=False)
    classify.init_model()
    classify.load_model(model_path)


def test_classification_model_lr3dcnn_predict(metaimage_path, model_path):
    classify = model.Model(init_model=False, batch_size=1)
    classify.init_model()
    classify.load_model(model_path)
    classify.train([{'file_path': metaimage_path,
                     'centroids': [{'x': 556, 'y': 30, 'z': -155, 'cancerous': True},
                                   {'x': 540, 'y': 20, 'z': -155, 'cancerous': False}]}])

    # assert len(predicted) == 1
    # assert isinstance(predicted[0]['p_concerning'], float)
    # assert predicted[0]['p_concerning'] >= 0.
    # assert predicted[0]['p_concerning'] <= 1.

# def test_classify_predict_inference(dicom_path, model_path):
#     params = preprocess_ct.Params(clip_lower=-1000,
#                                   clip_upper=400,
#                                   spacing=(.6, .6, .3))
#     preprocess = preprocess_ct.PreprocessCT(params)
#     predicted = trained_model.predict(dicom_path,
#                                       [{'x': 50, 'y': 50, 'z': 21}],
#                                       model_path,
#                                       preprocess_ct=preprocess,
#                                       preprocess_model_input=preprocess_LR3DCNN)
#
#     assert len(predicted) == 1
#     assert isinstance(predicted[0]['p_concerning'], float)
#     assert predicted[0]['p_concerning'] >= 0.
#     assert predicted[0]['p_concerning'] <= 1.
