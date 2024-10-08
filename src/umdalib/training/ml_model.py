try:
    from umdalib.utils import logger
except ImportError:
    from loguru import logger

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path as pt
from time import perf_counter
from typing import Dict, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost import __version__ as catboost_version
from dask.diagnostics import ProgressBar
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
from dask_ml.model_selection import RandomizedSearchCV as DaskRandomizedSearchCV
from joblib import __version__ as joblib_version
from joblib import dump, parallel_config
from lightgbm import LGBMRegressor
from lightgbm import __version__ as lightgbm_version
from scipy.optimize import curve_fit
from sklearn import __version__ as sklearn_version
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

# models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor

# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR

# for saving models
from sklearn.utils import resample
from tqdm import tqdm
from xgboost import XGBRegressor
from xgboost import __version__ as xgboost_version

from umdalib.training.read_data import read_as_ddf
from umdalib.utils import Paths

from umdalib.training.utils import get_transformed_data, Yscalers

tqdm.pandas()

logger.info(f"xgboost version {xgboost_version}")
logger.info(f"catboost version {catboost_version}")
logger.info(f"lightgbm version {lightgbm_version}")

logger.info(f"Using joblib version {joblib_version}")
logger.info(f"Using scikit-learn version {sklearn_version}")

# from dask.distributed import Client


# Set up Dask client
# client = Client()  # This will start a local cluster


def linear(x, m, c):
    return m * x + c


# models_dict
models_dict = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "svr": SVR,
    "knn": KNeighborsRegressor,
    "rfr": RandomForestRegressor,
    "gbr": GradientBoostingRegressor,
    "gpr": GaussianProcessRegressor,
    "xgboost": XGBRegressor,
    "catboost": CatBoostRegressor,
    "lgbm": LGBMRegressor,
}

n_jobs_keyword_available_models = ["linear_regression", "knn", "rfr", "xgboost", "lgbm"]

kernels_dict = {
    "Constant": kernels.ConstantKernel,
    "RBF": kernels.RBF,
    "Matern": kernels.Matern,
    "RationalQuadratic": kernels.RationalQuadratic,
    "ExpSineSquared": kernels.ExpSineSquared,
    "DotProduct": kernels.DotProduct,
    "WhiteKernel": kernels.WhiteKernel,
}

grid_search_dict = {
    "GridSearchCV": {"function": GridSearchCV, "parameters": []},
    "HalvingGridSearchCV": {"function": HalvingGridSearchCV, "parameters": ["factor"]},
    "RandomizedSearchCV": {"function": RandomizedSearchCV, "parameters": ["n_iter"]},
    "HalvingRandomSearchCV": {
        "function": HalvingRandomSearchCV,
        "parameters": ["factor"],
    },
    "DaskGridSearchCV": {"function": DaskGridSearchCV, "parameters": ["factor"]},
    "DaskRandomizedSearchCV": {
        "function": DaskRandomizedSearchCV,
        "parameters": ["n_iter"],
    },
}

random_state_supported_models = ["rfr", "gbr", "gpr"]

# seed = 2024
# rng = np.random.RandomState(seed)
rng = None


class TrainingFile(TypedDict):
    filename: str
    filetype: str
    key: str


@dataclass
class Args:
    model: str
    test_size: float
    bootstrap: bool
    bootstrap_nsamples: int
    parameters: Dict[str, Union[str, int, None]]
    fine_tuned_hyperparameters: Dict[str, Union[str, int, float, None]]
    pre_trained_file: str
    cv_fold: int
    cross_validation: bool
    training_file: TrainingFile
    training_column_name_y: str
    npartitions: int
    vectors_file: str
    noise_percentage: float
    ytransformation: str
    yscaling: str
    embedding: str
    pca: bool
    save_pretrained_model: bool
    fine_tune_model: bool
    grid_search_method: str
    grid_search_parameters: Dict[str, int]
    parallel_computation: bool
    n_jobs: int
    parallel_computation_backend: str
    use_dask: bool
    skip_invalid_y_values: bool
    inverse_scaling: bool
    inverse_transform: bool


def augment_data(
    X: np.ndarray, y: np.ndarray, n_samples: int, noise_percentage: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    augment_data a small dataset to create a larger training set.

    :X: Feature matrix
    :y: Target vector
    :n_samples: Number of samples in the bootstrapped dataset
    :noise_percentage: Scale of Gaussian noise to add to y
    :return: Bootstrapped X and y
    """

    logger.info(f"Augmenting data with {n_samples} samples")
    X_boot, y_boot = resample(X, y, n_samples=n_samples, replace=True, random_state=rng)

    logger.info(f"Adding noise percentage: {noise_percentage}")
    noise_scale = (noise_percentage / 100) * np.abs(y_boot)

    y_boot += np.random.normal(0, noise_scale)

    return X_boot, y_boot


def make_custom_kernels(kernel_dict: Dict[str, Dict[str, str]]) -> kernels.Kernel:
    constants_kernels = None
    other_kernels = None

    for kernel_key in kernel_dict.keys():
        kernel_params = kernel_dict[kernel_key]

        for kernel_params_key, kernel_params_value in kernel_params.items():
            if "," in kernel_params_value:
                kernel_params[kernel_params_key] = tuple(
                    float(x) for x in kernel_params_value.split(",")
                )
            elif kernel_params_value != "fixed":
                kernel_params[kernel_params_key] = float(kernel_params_value)
        logger.info(f"{kernel_key=}, {kernel_params=}")

        if kernel_key == "Constant":
            constants_kernels = kernels_dict[kernel_key](**kernel_params)
        else:
            if other_kernels is None:
                other_kernels = kernels_dict[kernel_key](**kernel_params)
            else:
                other_kernels += kernels_dict[kernel_key](**kernel_params)

    kernel = constants_kernels * other_kernels
    return kernel


def get_stats(estimator, X_true: np.ndarray, y_true: np.ndarray):
    y_pred: np.ndarray = estimator.predict(X_true)

    if inverse_scaling and yscaler:
        logger.info("Inverse transforming Y-data")
        y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true = yscaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    if inverse_transform and ytransformation:
        logger.info("Inverse transforming Y-data")
        if y_transformer:
            logger.info(
                f"Using Yeo-Johnson inverse transformation using {y_transformer=}"
            )
            y_pred = y_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_true = y_transformer.inverse_transform(y_true.reshape(-1, 1)).flatten()
        else:
            logger.info("Using other inverse transformation")
            y_true = get_transformed_data(
                y_true,
                ytransformation,
                inverse=True,
                lambda_param=boxcox_lambda_param,
            )
            y_pred = get_transformed_data(
                y_pred,
                ytransformation,
                inverse=True,
                lambda_param=boxcox_lambda_param,
            )

    logger.info("Evaluating model")
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)

    logger.info(f"R2: {r2:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    pop, _ = curve_fit(linear, y_true, y_pred)
    y_linear_fit = linear(y_true, *pop)
    y_linear_fit = np.array(y_linear_fit, dtype=float)

    return r2, mse, rmse, mae, y_true, y_pred, y_linear_fit


def save_intermediate_results(
    grid_search: GridSearchCV, filename: str = "intermediate_results.csv"
) -> None:
    """Saves intermediate cv_results_ to a CSV file."""
    df = pd.DataFrame(grid_search.cv_results_)
    df = df.sort_values(by="rank_test_score")
    df.to_csv(filename, index=False)


def compute(args: Args, X: np.ndarray, y: np.ndarray):
    start_time = perf_counter()

    estimator = None
    grid_search = None

    pre_trained_file = pt(args.pre_trained_file.strip()).with_suffix(".pkl")
    pre_trained_loc = pre_trained_file.parent
    if not pre_trained_loc.exists():
        pre_trained_loc.mkdir(parents=True)

    arguments_file = pre_trained_loc / f"{pre_trained_file.stem}.arguments.json"

    with open(arguments_file, "w") as f:
        json.dump(args.__dict__, f, indent=4)
        logger.info(f"Arguments saved to {arguments_file}")

    # bootstrap data
    if args.bootstrap:
        logger.info("Bootstrapping data")
        X, y = augment_data(
            X,
            y,
            n_samples=int(args.bootstrap_nsamples),
            noise_percentage=float(args.noise_percentage),
        )

    # stack the arrays (n_samples, n_features)
    if len(X.shape) == 1:
        logger.info("Reshaping X")
        X = np.vstack(X)

    logger.info(f"{X[0].shape=}\n{y[0]=}")
    logger.info(f"Loaded data: {X.shape=}, {y.shape=}")

    test_size = float(args.test_size)
    y_copy = y.copy()
    if test_size > 0:
        # split data
        logger.info("Splitting data for training and testing")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_copy,
            test_size=test_size,
            shuffle=True,
            # random_state=rng
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y_copy, y_copy

    logger.info(f"{X_train.shape=}, {X_test.shape=}")
    logger.info(f"{y_train.shape=}, {y_test.shape=}")

    if (
        args.model in random_state_supported_models
        and "random_state" not in args.parameters
        and rng is not None
    ):
        args.parameters["random_state"] = rng

    kernel = None
    if args.model == "gpr":
        logger.info("Using Gaussian Process Regressor with custom kernel")

        if "kernel" in args.parameters and args.parameters["kernel"]:
            kernel = make_custom_kernels(args.parameters["kernel"])
            args.parameters.pop("kernel", None)

    if args.model == "catboost":
        args.parameters["train_dir"] = str(Paths().app_log_dir / "catboost_info")
        logger.info(f"catboost_info: {args.parameters['train_dir']=}")

    logger.info(f"{models_dict[args.model]=}")

    if args.fine_tune_model:
        logger.info("Fine-tuning model")
        opts = {
            k: v
            for k, v in args.parameters.items()
            if k not in args.fine_tuned_hyperparameters.keys()
        }

        if args.parallel_computation and args.model in n_jobs_keyword_available_models:
            opts["n_jobs"] = n_jobs

        initial_estimator = models_dict[args.model](**opts)

        logger.info("Running grid search")
        # Grid-search
        # cv_fold = KFold(n_splits=int(args.cv_fold), shuffle=True, random_state=rng)
        # cv_fold = KFold(n_splits=int(args.cv_fold), shuffle=True)
        GridCV = grid_search_dict[args.grid_search_method]["function"]
        GridCV_parameters = {}
        for param in grid_search_dict[args.grid_search_method]["parameters"]:
            if param in args.grid_search_parameters:
                GridCV_parameters[param] = args.grid_search_parameters[param]

        if args.parallel_computation:
            GridCV_parameters["n_jobs"] = n_jobs

        logger.info(f"{GridCV=}, {GridCV_parameters=}")

        grid_search = GridCV(
            initial_estimator,
            args.fine_tuned_hyperparameters,
            cv=int(args.cv_fold),
            **GridCV_parameters,
        )
        logger.info("Fitting grid search")

        # run grid search
        grid_search.fit(X_train, y_train)
        estimator = grid_search.best_estimator_

        logger.info("Grid search complete")
        logger.info(f"Best score: {grid_search.best_score_}")
        logger.info(f"Best parameters: {grid_search.best_params_}")

        # client.close()

        # save grid search
        # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.save_pretrained_model:
            grid_savefile = pre_trained_file.with_name(
                f"{pre_trained_file.stem}_grid_search"
            ).with_suffix(".pkl")
            dump(grid_search, grid_savefile)

            df = pd.DataFrame(grid_search.cv_results_)
            df = df.sort_values(by="rank_test_score")
            df.to_csv(grid_savefile.with_suffix(".csv"))

            logger.info(f"Grid search saved to {grid_savefile}")
    else:
        if args.parallel_computation and args.model in n_jobs_keyword_available_models:
            args.parameters["n_jobs"] = n_jobs

        if args.model == "gpr" and kernel is not None:
            estimator = models_dict[args.model](kernel, **args.parameters)
        else:
            estimator = models_dict[args.model](**args.parameters)

    # estimator = Pipeline([("estimator", estimator)])

    # train model
    if not args.fine_tune_model:
        logger.info("Training model")
        estimator.fit(X_train, y_train)
        logger.info("Training complete")
    else:
        logger.info("Using best estimator from grid search")

    if args.save_pretrained_model:
        dump((estimator, yscaler), pre_trained_file)

    logger.info(f"Saving model to {pre_trained_file}")
    current_time = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")

    trained_params = estimator.get_params()
    if args.model == "catboost":
        logger.info(f"{estimator.get_all_params()=}")
        trained_params = trained_params | estimator.get_all_params()

    user_specified_params = args.parameters
    logger.info(
        f"{args.model=}, {user_specified_params=}\n{trained_params=}\n{yscaler=}"
    )

    parameters_file = pre_trained_file.with_suffix(".parameters.json")
    if args.save_pretrained_model:
        with open(parameters_file, "w") as f:
            parameters_dict = {
                "values": user_specified_params | trained_params,
                "model": args.model,
                "timestamp": current_time,
            }
            json.dump(parameters_dict, f, indent=4)
            logger.info(f"Model parameters saved to {parameters_file.name}")

    test_stats = get_stats(estimator, X_test, y_test)
    train_stats = get_stats(estimator, X_train, y_train)

    if inverse_scaling and yscaler:
        y = yscaler.inverse_transform(y.reshape(-1, 1)).flatten()
    if inverse_transform and ytransformation:
        if y_transformer:
            y = y_transformer.inverse_transform(y.reshape(-1, 1)).flatten()
        else:
            y = get_transformed_data(
                y, ytransformation, inverse=True, lambda_param=boxcox_lambda_param
            )

    results = {
        "data_shapes": {
            "X": X.shape,
            "y": y.shape,
            "X_test": X_test.shape,
            "y_test": y_test.shape,
            "X_train": X_train.shape,
            "y_train": y_train.shape,
        },
        "test_stats": {
            "r2": f"{test_stats[0]:.2f}",
            "mse": f"{test_stats[1]:.2f}",
            "rmse": f"{test_stats[2]:.2f}",
            "mae": f"{test_stats[3]:.2f}",
        },
        "train_stats": {
            "r2": f"{train_stats[0]:.2f}",
            "mse": f"{train_stats[1]:.2f}",
            "rmse": f"{train_stats[2]:.2f}",
            "mae": f"{train_stats[3]:.2f}",
        },
    }

    results["bootstrap"] = args.bootstrap
    if args.bootstrap:
        results["bootstrap_nsamples"] = args.bootstrap_nsamples
        results["noise_percentage"] = args.noise_percentage

    # if args.save_pretrained_model:
    with open(f"{pre_trained_file.with_suffix('.dat.json')}", "w") as f:
        json.dump(
            {
                "test": {
                    "y_true": test_stats[4].tolist(),
                    "y_pred": test_stats[5].tolist(),
                    "y_linear_fit": test_stats[6].tolist(),
                },
                "train": {
                    "y_true": train_stats[4].tolist(),
                    "y_pred": train_stats[5].tolist(),
                    "y_linear_fit": train_stats[6].tolist(),
                },
            },
            f,
            indent=4,
        )

    # Additional validation step
    results["cross_validation"] = args.cross_validation

    if args.cross_validation and not args.fine_tune_model and test_size > 0:
        logger.info("Cross-validating model")

        results["cv_fold"] = args.cv_fold

        cv_fold = KFold(n_splits=int(args.cv_fold), shuffle=True)
        cv_scores = cross_val_score(
            estimator, X, y, cv=cv_fold, scoring="r2", n_jobs=n_jobs
        )
        logger.info(f"Cross-validation R2 scores: {cv_scores}")
        logger.info(
            f"Mean CV R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )
        results["cv_scores"] = {
            "mean": f"{cv_scores.mean():.2f}",
            "std": f"{cv_scores.std() * 2:.2f}",
            "scores": cv_scores.tolist(),
        }

    if args.fine_tune_model:
        results["best_params"] = grid_search.best_params_
        results["best_score"] = f"{grid_search.best_score_:.2f}"

    results["timestamp"] = current_time

    end_time = perf_counter()
    logger.info(f"Training completed in {(end_time - start_time):.2f} s")
    results["time"] = f"{(end_time - start_time):.2f} s"

    with open(
        pre_trained_file.with_suffix(".results.json"),
        "w",
    ) as f:
        json.dump(results, f, indent=4)
        logger.info(f"Results saved to {pre_trained_file.with_suffix('.json')}")

    return results


def convert_to_float(value: Union[str, float]) -> float:
    try:
        return float(value)
    except ValueError:
        if isinstance(value, str) and "-" in value:
            parts = value.split("-")
            if len(parts) == 2 and parts[0] and parts[1]:
                try:
                    return (float(parts[0]) + float(parts[1])) / 2
                except ValueError:
                    pass
        if skip_invalid_y_values:
            return np.nan
        raise


n_jobs = None
backend = "threading"
skip_invalid_y_values = False
ytransformation: str = None
y_transformer = None
yscaling: str = "StandardScaler"
yscaler = None
boxcox_lambda_param = None

inverse_scaling = True
inverse_transform = True


def main(args: Args):
    global \
        n_jobs, \
        backend, \
        skip_invalid_y_values, \
        ytransformation, \
        yscaling, \
        yscaler, \
        boxcox_lambda_param, \
        inverse_scaling, \
        inverse_transform, \
        y_transformer

    ytransformation = args.ytransformation
    yscaling = args.yscaling
    inverse_scaling = args.inverse_scaling
    inverse_transform = args.inverse_transform

    skip_invalid_y_values = args.skip_invalid_y_values
    if args.parallel_computation:
        n_jobs = int(args.n_jobs)
        backend = args.parallel_computation_backend

    logger.info(f"Training {args.model} model")
    logger.info(f"{args.training_file['filename']}")

    X = np.load(args.vectors_file, allow_pickle=True)

    # load training data from file
    ddf = read_as_ddf(
        args.training_file["filetype"],
        args.training_file["filename"],
        args.training_file["key"],
        use_dask=args.use_dask,
    )

    y = None
    if args.use_dask:
        ddf = ddf.repartition(npartitions=args.npartitions)
        with ProgressBar():
            y = ddf[args.training_column_name_y].compute()
    else:
        y = ddf[args.training_column_name_y]
    logger.info(f"{type(y)=}")

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Apply the conversion function to handle strings like '188.0 - 189.0'
    y = y.apply(convert_to_float)

    # Keep track of valid indices
    valid_y_indices = y.notna()
    y = y[valid_y_indices]
    X = X[valid_y_indices]

    y = y.values

    invalid_embedding_indices = [i for i, arr in enumerate(X) if np.any(arr == 0)]

    # Initially, mark all as valid
    valid_embedding_mask = np.ones(len(X), dtype=bool)
    # Then, mark invalid indices as False
    valid_embedding_mask[invalid_embedding_indices] = False

    X = X[
        valid_embedding_mask
    ]  # Keep only the rows that are marked as True in the valid_embedding_mask
    y = y[valid_embedding_mask]

    y_transformer = None

    if ytransformation:
        if ytransformation == "boxcox":
            logger.info("Applying boxcox transformation")
            y, boxcox_lambda_param = get_transformed_data(
                y, ytransformation, get_other_params=True
            )
            logger.info(f"{boxcox_lambda_param=}")
        elif ytransformation == "yeo_johnson":
            logger.info("Applying yeo-johnson transformation")
            y, y_transformer = get_transformed_data(
                y, ytransformation, get_other_params=True
            )
            logger.info(f"{y_transformer=}")
        else:
            logger.info(f"Applying {ytransformation} transformation")
            y = get_transformed_data(y, ytransformation)
    else:
        logger.warning("No transformation applied")

    yscaler = None
    if yscaling:
        yscaler = Yscalers[yscaling]()
        y = yscaler.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        logger.warning("No scaling applied")

    logger.info(f"{y[:5]=}, {type(y)=}")

    results = None
    if args.parallel_computation:
        with parallel_config(backend, n_jobs=n_jobs):
            logger.info(f"Using {n_jobs} jobs with {backend} backend")
            results = compute(args, X, y)
    else:
        logger.info("Running in serial mode")
        results = compute(args, X, y)

    return results
