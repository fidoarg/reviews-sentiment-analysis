from lightgbm import LGBMClassifier


def create_model(**kwargs):

    model = LGBMClassifier(**kwargs)
    return model