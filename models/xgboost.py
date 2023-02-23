from xgboost import XGBClassifier


def create_model(**kwargs):
    
    model = XGBClassifier(**kwargs)
    return model
