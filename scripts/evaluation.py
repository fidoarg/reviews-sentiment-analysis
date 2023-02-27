import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import label_binarize


def get_performance(predictions, y_test, labels=[1, 0]):
    # Put your code
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
    precision = metrics.precision_score(y_true=y_test, y_pred=predictions)
    recall = metrics.recall_score(y_true=y_test, y_pred=predictions)
    f1_score = metrics.f1_score(y_true=y_test, y_pred=predictions)

    report = metrics.classification_report(
        y_true=y_test, y_pred=predictions, labels=labels)

    cm = metrics.confusion_matrix(
        y_true=y_test, y_pred=predictions, labels=labels)
    cm_as_dataframe = pd.DataFrame(data=cm)

    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)

    return accuracy, precision, recall, f1_score


def plot_roc(model, y_test, features):
    # Put your code
    # predictions= model.
    y_score = model.predict_proba(features)
    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, y_score[:, 1], pos_label=1)
    roc_auc = metrics.roc_auc_score(y_true=y_test, y_score=y_score)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

def get_performance_report(predictions, y_test, labels=[1, 0]):

    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
    precision = metrics.precision_score(y_true=y_test, y_pred=predictions)
    recall = metrics.recall_score(y_true=y_test, y_pred=predictions)
    f1_score = metrics.f1_score(y_true=y_test, y_pred=predictions)

    report = metrics.classification_report(
        y_true=y_test, y_pred=predictions, labels=labels)

    cm = metrics.confusion_matrix(
        y_true=y_test, y_pred=predictions, labels=labels)
    cm_as_dataframe = pd.DataFrame(data=cm)

    performance_report = 'Model Performance metrics:' + '\n' + \
        '-'*30 + '\n' + \
        'Accuracy: ' + str(round(accuracy, 4)) + '\n' + \
        'Precision: '+ str(round(precision, 4)) + '\n' + \
        'Recall: ' +  str(round(recall, 4)) + '\n' + \
        'F1 Score: ' + str(round(f1_score, 4)) + '\n' + \
        '\nModel Classification report: ' + '\n' + \
        '-'*30 + '\n' + \
        report + '\n' + \
        '\nPrediction Confusion Matrix:' + '\n' + \
        '-'*30 + '\n' + \
        str(cm_as_dataframe)

    return performance_report