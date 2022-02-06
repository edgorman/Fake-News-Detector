"""
    main.py - Edward Gorman - eg6g17@soton.ac.uk
"""
import os
import numpy as np
from scipy.stats import stats
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.ensemble import AdaBoostClassifier
from process_data import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_DIRECTORY = os.path.join(os.getcwd(), "data")
TESTING_FILE = "testset.txt"
TRAINING_FILE = "trainingset.txt"


def train_model(classifier, features, labels):
    return classifier.fit(features, labels)


def test_model(classifier, features, labels):
    preds = classifier.predict(features)
    print("\taccuracy:",    round(accuracy_score(labels, preds),  2),
          "\tprecision:",   round(precision_score(labels, preds), 2),
          "\trecall:",      round(precision_score(labels, preds), 2),
          "\tf1 score:",    round(f1_score(labels, preds, pos_label=1, average='binary'), 2))
    return preds


if __name__ == "__main__":
    # Load data from csv
    print("Loading data . . .")
    testing = load_data(os.path.join(DATA_DIRECTORY, TESTING_FILE))
    training = load_data(os.path.join(DATA_DIRECTORY, TRAINING_FILE))

    # Create models
    models = {
        "text_length": AdaBoostClassifier(n_estimators=500, learning_rate=1.0, algorithm='SAMME.R'),
        "timestamp_feats": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, algorithm='SAMME'),
        "text_tfidf": svm.SVC(C=0.6, kernel='rbf', gamma='scale'),
    }

    # Train models
    print("Training models . . .")
    tl_features, tl_labels = extract_text_length(training.copy())
    train_model(models["text_length"], tl_features, tl_labels)
    print("\t Finished text_length")
    ts_features, ts_labels = extract_timestamp_feats(training.copy())
    train_model(models["timestamp_feats"], ts_features, ts_labels)
    print("\t Finished timestamp_feats")
    tfidf_vect = get_tfidf_vect(training.copy())
    tf_features, tf_labels = extract_text_tfidf(training.copy(), tfidf_vect)
    train_model(models["text_tfidf"], tf_features, tf_labels)
    print("\t Finished text_tfidf")

    # Create predictions
    predictions = {}

    # Test models
    print("Testing models . . .")
    tl_features, tl_labels = extract_text_length(testing.copy())
    predictions["text_length"] = test_model(models["text_length"], tl_features, tl_labels)
    ts_features, ts_labels = extract_timestamp_feats(testing.copy())
    predictions["timestamp_feats"] = test_model(models["timestamp_feats"], ts_features, ts_labels)
    tf_features, tf_labels = extract_text_tfidf(testing.copy(), tfidf_vect)
    predictions["text_tfidf"] = test_model(models["text_tfidf"], tf_features, tf_labels)

    # Combine models
    print("Combining models . . .")
    predictions_array = np.array([predictions["text_length"],
                                  predictions["timestamp_feats"],
                                  predictions["text_tfidf"]])
    votes = stats.mode(predictions_array)[0][0]
    print("\taccuracy:", round(accuracy_score(testing['label'], votes), 2),
          "\tprecision:", round(precision_score(testing['label'], votes), 2),
          "\trecall:", round(precision_score(testing['label'], votes), 2),
          "\tf1 score:", round(f1_score(testing['label'], votes, pos_label=1, average='binary'), 2))
