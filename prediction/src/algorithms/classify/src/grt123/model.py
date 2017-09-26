"""
Classification model from team gtr123
part of the Winning algorithm for DSB2017

code was adopted from https://github.com/lfz/DSB2017
"""
import numpy as np
import torch
from src.preprocess import preprocess_ct, load_ct, crop_patches, generators
from torch.autograd import Variable

from prediction.src.algorithms.classify.src.classification_model import ClassificationModel
from .architecture import CaseNet
from .configuration import config


class Model(ClassificationModel):
    def __init__(self, init_model=True, pull_size=10, batch_size=32, data_format=None):
        super(Model, self).__init__(init_model, pull_size, batch_size, data_format)

        self.config = config
        self.data_generator = generators.DataGenerator(featurewise_center=True,
                                                       featurewise_std_normalization=True,
                                                       rotation_range=20,
                                                       shift_range=.1,
                                                       zoom_lower=.8,
                                                       zoom_upper=1.2,
                                                       zoom_independent=False,
                                                       fill_mode='nearest',
                                                       preprocessing_function=None)

    def init_model(self):
        pass

    def _ct_preprocess(self, ct_path):
        ct_array, meta = load_ct.load_ct(ct_path)
        process = preprocess_ct.PreprocessCT(hu_transform=True,
                                             clip_lower=-1000,
                                             clip_upper=400,
                                             min_max_normalize=False)
        return process(ct_array, meta)

    def _batch_process(self, batch, labels=None):
        generator = self.data_generator.flow(batch, labels, batch_size=self.batch_size, shuffle=False)
        batch, labels = next(generator)
        # batch = np.rollaxis(batch, self.channel_axis, 0)
        # pads = np.array([(12, 21, 21),
        #                  (21, 12, 21),
        #                  (21, 21, 12)])
        # centroid = np.array(batch.shape[2:]) // 2
        #
        # batch = [batch[:, :,
        #                centroid[0] - pad[0]: centroid[0] + pad[0],
        #                centroid[1] - pad[1]: centroid[1] + pad[1],
        #                centroid[2] - pad[2]: centroid[2] + pad[2]]
        #          for pad in pads]
        # batch = [np.rollaxis(view, 0, self.channel_axis + 1) for view in batch]
        if labels is not None:
            labels = np.array([[1, 0] if label else [0, 1] for label in labels])
        return batch, labels

    @staticmethod
    def _sample_annotations(annotations, sampling_pure, sampling_cancerous):
        """Train the model through the annotated CT scans

        Args:
            annotations (list[dict]): A list of centroids of the form::
                 {'file_path': str,
                  'centroids': [{'x': int,
                                 'y': int,
                                 'z': int,
                                 'cancerous': bool}, ..]}.
            sampling_pure (float): coefficient of .
            sampling_cancerous (float): .

        Returns:
            list[dict]: list of sampled centroids of the same form as annotations::
                 {'file_path': str,
                  'centroids': [{'x': int,
                                 'y': int,
                                 'z': int,
                                 'cancerous': bool}, ..]}.
        """
        cancerous = [(patient['file_path'], finding)
                     for patient in annotations
                     for finding in patient['centroids']
                     if finding['cancerous']]
        pure = [(patient['file_path'], finding)
                for patient in annotations
                for finding in patient['centroids']
                if not finding['cancerous']]
        coeff = int(np.ceil(sampling_cancerous * len(cancerous)))
        sampled = [cancerous[idx] for idx in np.random.choice(len(cancerous), coeff)]

        coeff = int(np.ceil(sampling_pure * len(pure)))
        sampled.extend([pure[idx] for idx in np.random.choice(len(pure), coeff)])
        np.random.shuffle(sampled)
        resulted = defaultdict(list)
        for file_path, sample in sampled:
            resulted[file_path] += [sample]
        return [{'file_path': key, 'centroids': el} for key, el in resulted.items()]

    def on_epoch_start(self, annotations, sampling_pure, sampling_cancerous):
        np.random.shuffle(annotations)
        sampled = self._sample_annotations(annotations, sampling_pure, sampling_cancerous)
        return sampled

    def feed(self, annotations, sampling_pure, sampling_cancerous):
        """Train the model through the annotated CT scans

                Args:
                    annotations (list[dict]): A list of centroids of the form::
                         {'file_path': str,
                          'centroids': [{'x': int,
                                         'y': int,
                                         'z': int,
                                         'cancerous': bool}, ..]}.
                    sampling_pure (float): coefficient of .
                    sampling_cancerous (float): .

                Yields:
                    list[np.ndarray]: list of patches.
        """
        while True:
            sampled = self.on_epoch_start(annotations, sampling_pure, sampling_cancerous)
            findings_amount = sum([len(patient['centroids']) for patient in sampled])
            iterations = int(np.ceil(findings_amount / self.pull_ct.maxlen))
            for counter in range(iterations):
                batch = sampled[counter * self.pull_ct.maxlen: (counter + 1) * self.pull_ct.maxlen]
                for patient in batch:
                    ct_array, meta = self._ct_preprocess(patient['file_path'])
                    self.pull_ct.append((patient['centroids'], ct_array, meta))
                while self.pull_ct:
                    centroids, ct_array, meta = self.pull_ct.pop()
                    patches = crop_patches.patches_from_ct(ct_array, centroids, patch_shape=(42, 42, 42), meta=meta)
                    patches = [(centroid['cancerous'], patch) for centroid, patch in zip(centroids, patches)]
                    self.pull_patches.extend(patches)

                np.random.shuffle(self.pull_patches)
                allowed_iterations = len(self.pull_patches) // self.batch_size

                for j in range(allowed_iterations):
                    batch = list()
                    for i in range(self.batch_size):
                        batch.append(self.pull_patches.pop())
                    print(batch[-1][1].shape)
                    batch, labels = self._batch_process(np.expand_dims(np.asarray([patch for _, patch in batch]),
                                                                       self.channel_axis),
                                                        [label for label, _ in batch])
                    yield batch, labels

    def train(self, annotations, validation_data=None, epochs=1, model_path=None, verbose=False):
        """Train the model through the annotated CT scans

        Args:
            annotations (list[dict]): A list of centroids of the form::
                 {'file_path': str,
                  'centroids': [{'x': int,
                                 'y': int,
                                 'z': int,
                                 'cancerous': bool}, ..]}.
            model_path (str): a path where to store the serialized model.
            epochs (int): amount of epochs to be trained
            validation_data (float | list[dict]): data for validation.
                If float (should be in the range [0, 1]), then the corresponding
                portion of training data will be excluded for the validation purpose.
            verbose (bool): whether to print the training progress into std:out.


        Returns:
            tf.model: a model trained over annotated data.
        """
        valid_generator = None
        validation_steps = None
        if isinstance(validation_data, float):
            valid_size = int(len(annotations) * validation_data)
            np.random.shuffle(annotations)
            validation_data = annotations[:valid_size]
            annotations = annotations[valid_size:]

        if validation_data:
            validation_steps = sum([len(patient['centroids']) for patient in validation_data])
            valid_generator = self.feed(validation_data,
                                        sampling_pure=1,
                                        sampling_cancerous=1)

        sampled = self._sample_annotations(annotations,
                                           sampling_pure=.04,
                                           sampling_cancerous=.7)
        findings_amount = sum([len(patient['centroids']) for patient in sampled])
        train_generator = self.feed(annotations, sampling_pure=.04, sampling_cancerous=.7)
        callback = None
        if model_path:
            callback = [keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=verbose, save_best_only=True)]
        a, l = next(train_generator)
        print(l)
        self.model.fit_generator(train_generator, findings_amount, epochs=epochs, verbose=verbose, callbacks=callback,
                                 validation_data=valid_generator, validation_steps=validation_steps)
        return self.model

    def load_model(self, model_path):
        casenet = CaseNet()
        casenet.load_state_dict(torch.load(model_path))
        casenet.eval()

        if torch.cuda.is_available():
            casenet = torch.nn.DataParallel(casenet).cuda()
        # else:
            # casenet = torch.nn.parallel.DistributedDataParallel(casenet)
        self.model = casenet

    def predict(self, candidates, model_path=None):
        """ Predict cancerous of given candidates.

        Args:
            candidates (list[dict]): A list of centroids of the form::
                {'file_path': str,
                 'centroids': {'x': int,
                               'y': int,
                               'z': int}}
            model_path (str): A path to the serialized model


        Returns:
            (list[dict]): A list of centroids of the form::
                {'file_path': str,
                 'centroids': {'x': int,
                               'y': int,
                               'z': int,
                               'p_concerning': float}}
        """
        if model_path is not None:
            self.load_model(model_path)

        if not candidates:
            return []

        preprocess = preprocess_ct.PreprocessCT(clip_lower=-1200., clip_upper=600., spacing=1., order=1,
                                                min_max_normalize=True, scale=255, dtype='uint8')
        ct_array, meta = preprocess(*load_ct.load_ct(ct_path))
        patches = crop_patches.patches_from_ct(ct_array, meta, config['crop_size'], nodule_list,
                                               stride=config['stride'], pad_value=config['filling_value'])
        results = []
        for nodule, (cropped_image, coords) in zip(nodule_list, patches):
            cropped_image = Variable(torch.from_numpy(cropped_image[np.newaxis, np.newaxis]).float())
            cropped_image.volatile = True
            coords = Variable(torch.from_numpy(coords[np.newaxis]).float())
            coords.volatile = True
            _, pred, _ = self.model(cropped_image, coords)
            results.append(
                {"x": nodule["x"], "y": nodule["y"], "z": nodule["z"], "p_concerning": float(pred.data.cpu().numpy())})

    return results
