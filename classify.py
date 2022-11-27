import pickle
from prepare_data import Data
from sklearn import metrics


class Classification:
    def __init__(self):
        pass

    def classify(self, model):
        print('Reading test files....')
        data_model = Data()
        testing_data = data_model.generate_sfts_data(
            for_dataset='TEST')

        print('Predicting on the test data....')
        y_pred = model.predict(testing_data)
        print(y_pred)


if __name__ == '__main__':
    # load model
    with open('./trained-model.pkl', 'rb') as f:
        clf = pickle.load(f)

    classification_instance = Classification()
    classification_instance.classify(clf)
