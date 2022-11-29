import pickle
from prepare_data import Data
import numpy as np

class Classification:
    def __init__(self):
        print('Reading test files....')
        data_model = Data()
        self.testing_data = data_model.generate_sfts_data(
            for_dataset='TEST')

    def classify(self, model):
        print('Preparing data....')
        # make all sfts of the dimension of the fitted model
        max_sft_len = model.n_features_in_
        for i, d in enumerate(self.testing_data):
            self.testing_data[i] = np.pad(d, (0, max_sft_len - len(d)), 'median')

        print('Predicting on the test data....')
        y_pred = model.predict(self.testing_data)
        print(y_pred)


if __name__ == '__main__':
    # load models
    with open('./trained-model-svm.pkl', 'rb') as f:
        clf_svm = pickle.load(f)
    with open('./trained-model-dt.pkl', 'rb') as f:
        clf_dt = pickle.load(f)

    classification_instance = Classification()

    print('Classifying using svm classifier.....')
    classification_instance.classify(clf_svm)

    print('Classifying using dt classifier.....')
    classification_instance.classify(clf_dt)
