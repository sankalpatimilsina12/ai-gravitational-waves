from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pickle
from prepare_data import Data


class Model:
    def __init__(self):
        pass

    def generate_model(self):
        print('Reading training files....')
        data_model = Data()
        training_data, classification_labels = data_model.generate_sfts_data(
            for_dataset='TRAIN')

        X_train, X_test, y_train, y_test = train_test_split(
            training_data, classification_labels, test_size=0.20)

        print('Creating model....')
        clf = svm.SVC(kernel='linear')
        clf.fit(list(X_train), y_train)

        print('Predicting on test data....')
        y_pred = clf.predict(X_test)
        print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

        print('Saving the generated model....')
        with open('./trained-model.pkl', 'wb') as f:
            pickle.dump(clf, f)

        return clf


if __name__ == '__main__':
    model = Model()
    clf = model.generate_model()