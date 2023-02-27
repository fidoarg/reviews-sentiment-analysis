import argparse
import os
import pickle
import warnings

from models import lightgbm, randomforest, xgboost
from scripts.get_data import get_model_raw_data
from scripts.text_normalizing import TextNormalizer, NormTechniques
from scripts.text_vectorizing import TextVectorizer
from scripts.evaluation import get_performance_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from utils import utils
from textwrap import dedent

warnings.filterwarnings("ignore", category=UserWarning,
                        message="MarkupResemblesLocatorWarning")


def parse_text_normalizing(**params_text_norm: dict) -> dict:
    norm_techniques = params_text_norm['norm_to_use']
    if norm_techniques == 'all':
        function_output = {'norm_to_use': [
            NormTechniques.HTML_STRIPPING,
            NormTechniques.ACCENTED_CHAR_REMOVAL,
            NormTechniques.CONTRACTION_EXPANSION,
            NormTechniques.TEXT_LEMMATIZATION,
            NormTechniques.TEXT_STEMMING,
            NormTechniques.SPECIAL_CHAR_REMOVAL,
            NormTechniques.REMOVE_DIGITS,
            NormTechniques.STOPWORD_REMOVAL
        ]}
    elif isinstance(norm_techniques, list):
        function_output = {'norm_to_use': [
            member for member in NormTechniques if member.value in norm_techniques]}

    return function_output


def parse_text_vectorizing(**params_text_vect: dict) -> dict:
    if  params_text_vect.get('w2vec') == True:
        return {'kind': 'bow', 'w2vec': True}
    elif params_text_vect.get('kind') is not None and params_text_vect['kind'] in ('bow', 'tfidf'):
        return {'kind': params_text_vect['kind'], 'w2vec': False}
    elif params_text_vect.get('kind') is None:
        AttributeError('Vectorization kind missing')
    elif not params_text_vect['kind'] in ('bow', 'tfidf'):
        ValueError('Invalid vectorization kind')


def parse_args():
    """
    Use argparse to get the input parameters for training the model.
    """
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Full path to experiment configuration file.",
    )

    args = parser.parse_args()

    return args


def main(config_file):
    """
    Code for the training logic.
    Parameters
    ----------
    config_file : str
        Full path to experiment configuration file.
    """
    # Load configuration file, use utils.load_config()
    config = utils.load_config(config_file)
    # Load training dataset
    # We will split train data in train/validation while training our
    # model, keeping away from our experiments the testing dataset
    X_train, y_train = get_model_raw_data(
        config["data"]['train'],
    )
    X_test, y_test = get_model_raw_data(
        config["data"]['test'],
    )
    print(dedent(
        f"""
        ----------------------------
          DATA LOADED SUCCESSFULLY
        ----------------------------"""
    ))

    text_normalizing = parse_text_normalizing(**config['text_normalizing'])
    text_normalizer = TextNormalizer(**text_normalizing)

    text_vectorizing = parse_text_vectorizing(**config['text_vectorizing'])
    text_vectorizer = TextVectorizer(**text_vectorizing)

    model_type = config['model']['type']
    if model_type is None or not model_type in ('randomforest', 'xgboost', 'lightgbm'):
        raise ValueError('Invalid model type')

    if model_type == 'randomforest':
        model = randomforest.create_model(**config['model']["parameters"])
    elif model_type == 'xgboost':
        model = xgboost.create_model(**config['model']["parameters"])
    elif model_type == 'lightgbm':
        model = lightgbm.create_model(**config['model']["parameters"])

    experiment_dir = os.path.dirname(config_file)
    path_to_preprocesser = os.path.join(
        experiment_dir,
        f'data-preprocesser.pkl'
    )
    if os.path.exists(path_to_preprocesser):
        with open(path_to_preprocesser, 'rb') as preprocesser:
            preprocess_pipeline = pickle.load(preprocesser)
        X_train_model = preprocess_pipeline.transform(X=X_train)
    else:
        preprocess_pipeline = Pipeline(
            steps=[
                ('text_normalizer', text_normalizer),
                ('text_vectorizer', text_vectorizer),
            ]
        )
        X_train_model = preprocess_pipeline.fit_transform(X=X_train)

    with open(path_to_preprocesser, 'wb') as preprocesser_file:
        pickle.dump(preprocess_pipeline, preprocesser_file)

    print(dedent(
        f"""
        ----------------------------
         PREPROCESS PIPELINE CREATED
        ----------------------------"""
    ))

    param_grid = config['model']['param_grid']
    n_iter = config['train']['n_iter']
    cv = config['train']['cv']

    if config['model']['search_type'] == 'grid':
        grid_search = GridSearchCV(
            model, param_grid=param_grid, cv=cv, scoring='roc_auc', verbose=10)
    elif config['model']['search_type'] == 'random':
        grid_search = RandomizedSearchCV(
            model, param_distributions=param_grid, cv=cv,  n_iter=n_iter, scoring='roc_auc', verbose=10)

    # train the model using the GridSearchCV object

    grid_search.fit(X_train_model, y_train)
    # Update the progress bar for each fit
    print(dedent(
        f"""
        ------------------------
           TRAINING COMPLETED
        ------------------------"""
    ))

    best_model = grid_search.best_estimator_
    X_test_model = preprocess_pipeline.transform(X_test)
    y_pred = best_model.predict(X=X_test_model)
    y_probs = best_model.predict_proba(X=X_test_model)

    roc_auc = roc_auc_score(
        y_score=y_probs[:, 1], y_true=y_test)
    perf_report = get_performance_report(predictions=y_pred, y_test=y_test)

    path_to_report_file = os.path.join(
        experiment_dir,
        'experiment_report.txt'
    )

    path_to_model_file = os.path.join(
        experiment_dir,
        f'model-roc-auc-{roc_auc:.4f}.pkl'
    )

    with open(path_to_report_file, 'w') as report_file:
        report_file.write(perf_report)

    with open(path_to_model_file, 'wb') as model_file:
        pickle.dump(best_model, model_file)


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file)
