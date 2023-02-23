from sklearn.ensemble import RandomForestClassifier


def create_model(**kwargs):

    model = RandomForestClassifier(**kwargs)
    return model
