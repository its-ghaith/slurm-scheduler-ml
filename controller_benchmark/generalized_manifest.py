from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from controller_benchmark.manifest import validate_manifest


def _case(
    *,
    case_id: str,
    task_type: str,
    task_family: str,
    dataset_key: str,
    dataset_name: str,
    model_version: str,
    primary_metric: str,
    transform: str,
    scenario: str,
    flops: float,
    scale: float = 1.0,
    **options: Any,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "runner": "cross_domain_task",
        "task_type": task_type,
        "task_family": task_family,
        "quality_metric": "quality_score",
        "primary_metric": primary_metric,
        "quality_definition": {
            "metric": primary_metric,
            "transform": transform,
            "scale": scale,
        },
        "scenario": scenario,
        "training_seed": 0,
        "model_version": model_version,
        "model_flops_estimate": flops,
        "pretrained": False,
        "dataset_name": dataset_name,
        "dataset_key": dataset_key,
        "dataset_fingerprint": f"{dataset_key}-official-cache-v1",
        "runner_version": "cross-domain-task-v1",
        "progress_unit": "epoch",
        **options,
    }


def _paired_cases(
    task: dict[str, Any],
    *,
    checkpoint_root: str = "/workspace-cache/controller-pretrained-cross-domain",
) -> list[dict[str, Any]]:
    datasets = task["datasets"]
    cases: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets):
        scratch_id = f"{dataset['id']}-{task['case_suffix']}-scratch"
        source = datasets[(index + 1) % len(datasets)]
        source_id = f"{source['id']}-{task['case_suffix']}-scratch"
        common = dict(task.get("options", {}))
        common.update(dataset.get("options", {}))
        scratch = _case(
            case_id=scratch_id,
            task_type=task["task_type"],
            task_family=task["task_family"],
            dataset_key=dataset["key"],
            dataset_name=dataset["name"],
            model_version=task["model"],
            primary_metric=task["metric"],
            transform=task["transform"],
            scenario=f"{dataset['id']}-{task['model']}-scratch",
            flops=task["flops"],
            initialization="random initialization",
            **common,
        )
        pretrained = copy.deepcopy(scratch)
        pretrained.update(
            {
                "id": f"{dataset['id']}-{task['case_suffix']}-pretrained",
                "scenario": f"{dataset['id']}-{task['model']}-source-domain-pretrained",
                "pretrained": True,
                "initialization": "source-domain checkpoint from another dataset in the same task family",
                "pretrained_source_case_id": source_id,
                "pretrained_checkpoint": f"{checkpoint_root}/{source_id}.pt",
            }
        )
        cases.extend((scratch, pretrained))
    return cases


TASK_DEFINITIONS = [
    {
        "id": "tabular_classification",
        "title": "Tabular Classification",
        "task_type": "tabular_classification",
        "task_family": "tabular",
        "case_suffix": "tabular-classification",
        "model": "mlp-128",
        "metric": "macro_f1",
        "transform": "identity",
        "flops": 2.0e6,
        "options": {"batch_size": 256, "lcpfn_alignment": "LCBench modality and MLP family"},
        "datasets": [
            {"id": "adult", "key": "adult", "name": "UCI Adult"},
            {"id": "covertype", "key": "covertype", "name": "Forest Covertype"},
            {"id": "bank-marketing", "key": "bank_marketing", "name": "UCI Bank Marketing"},
        ],
    },
    {
        "id": "tabular_regression",
        "title": "Tabular Regression",
        "task_type": "tabular_regression",
        "task_family": "tabular",
        "case_suffix": "tabular-regression",
        "model": "mlp-regressor-128",
        "metric": "nrmse",
        "transform": "inverse_positive",
        "flops": 1.0e6,
        "options": {"batch_size": 256},
        "datasets": [
            {"id": "california-housing", "key": "california_housing", "name": "California Housing"},
            {"id": "diabetes", "key": "diabetes", "name": "Diabetes Regression"},
            {"id": "bike-sharing", "key": "bike_sharing", "name": "UCI Bike Sharing"},
        ],
    },
    {
        "id": "anomaly_detection",
        "title": "Anomaly Detection",
        "task_type": "anomaly_detection",
        "task_family": "tabular",
        "case_suffix": "anomaly-detection",
        "model": "autoencoder-128",
        "metric": "auroc",
        "transform": "identity",
        "flops": 4.0e6,
        "options": {"batch_size": 512},
        "datasets": [
            {"id": "kddcup99", "key": "kddcup99", "name": "KDD Cup 1999 SA"},
            {"id": "breast-cancer", "key": "breast_cancer_anomaly", "name": "Breast Cancer Wisconsin Anomaly"},
            {"id": "wine", "key": "wine_anomaly", "name": "Wine Anomaly"},
        ],
    },
    {
        "id": "text_classification",
        "title": "Text Classification",
        "task_type": "text_classification",
        "task_family": "text_classification",
        "case_suffix": "text-classification",
        "model": "bidirectional-gru-hash20k",
        "metric": "macro_f1",
        "transform": "identity",
        "flops": 5.0e8,
        "options": {"batch_size": 64, "max_train_samples": 10000, "validation_samples": 2500, "max_test_samples": 5000, "sequence_length": 256, "vocabulary_size": 20000},
        "datasets": [
            {"id": "imdb", "key": "imdb", "name": "IMDB Reviews", "options": {"lcpfn_alignment": "Taskset IMDB sentiment and RNN family"}},
            {"id": "ag-news", "key": "ag_news", "name": "AG News", "options": {"num_classes": 4}},
            {"id": "dbpedia", "key": "dbpedia", "name": "DBpedia Ontology", "options": {"num_classes": 14}},
        ],
    },
    {
        "id": "language_modeling",
        "title": "Language Modeling",
        "task_type": "language_modeling",
        "task_family": "language_modeling",
        "case_suffix": "language-modeling",
        "model": "transformer-encoder-lm-hash12k",
        "metric": "cross_entropy",
        "transform": "exp_negative",
        "flops": 2.0e9,
        "options": {"batch_size": 32, "sequence_length": 64, "vocabulary_size": 12000, "max_train_tokens": 300000},
        "datasets": [
            {"id": "wikitext2", "key": "wikitext2", "name": "WikiText-2", "options": {"lcpfn_alignment": "PD1 LM1B Transformer proxy"}},
            {"id": "penn-treebank", "key": "penn_treebank", "name": "Penn Treebank"},
            {"id": "tiny-shakespeare", "key": "tiny_shakespeare", "name": "Tiny Shakespeare"},
        ],
    },
    {
        "id": "time_series_forecasting",
        "title": "Time-Series Forecasting",
        "task_type": "time_series_forecasting",
        "task_family": "time_series_forecasting",
        "case_suffix": "time-series-forecasting",
        "model": "lstm-forecaster",
        "metric": "nrmse",
        "transform": "inverse_positive",
        "flops": 3.0e8,
        "options": {"batch_size": 128, "sequence_length": 96},
        "datasets": [
            {"id": "etth1", "key": "etth1", "name": "ETTh1"},
            {"id": "etth2", "key": "etth2", "name": "ETTh2"},
            {"id": "ettm1", "key": "ettm1", "name": "ETTm1"},
        ],
    },
    {
        "id": "recommendation",
        "title": "Recommendation",
        "task_type": "recommendation",
        "task_family": "recommendation",
        "case_suffix": "recommendation",
        "model": "hashed-matrix-factorization",
        "metric": "rmse",
        "transform": "inverse_positive",
        "flops": 1.0e7,
        "options": {"batch_size": 1024, "user_buckets": 8192, "item_buckets": 16384, "max_samples": 250000},
        "datasets": [
            {"id": "movielens100k", "key": "movielens100k", "name": "MovieLens 100K"},
            {"id": "movielens1m", "key": "movielens1m", "name": "MovieLens 1M"},
            {"id": "movielens10m", "key": "movielens10m", "name": "MovieLens 10M"},
        ],
    },
    {
        "id": "reinforcement_learning",
        "title": "Reinforcement Learning",
        "task_type": "reinforcement_learning",
        "task_family": "reinforcement_learning",
        "case_suffix": "reinforcement-learning",
        "model": "padded-state-dqn",
        "metric": "normalized_eval_reward",
        "transform": "identity",
        "flops": 1.0e6,
        "options": {"progress_unit": "evaluation_cycle", "episodes_per_checkpoint": 5, "evaluation_episodes": 10},
        "datasets": [
            {"id": "cartpole", "key": "cartpole", "name": "CartPole Standard"},
            {"id": "cartpole-heavy", "key": "cartpole_heavy", "name": "CartPole Heavy Pole"},
            {"id": "mountaincar", "key": "mountaincar", "name": "MountainCar"},
        ],
    },
]


HARD_TASK_DEFINITIONS = [
    {
        "id": "tabular_classification",
        "title": "Hard Tabular Classification",
        "task_type": "tabular_classification",
        "task_family": "tabular",
        "case_suffix": "tabular-classification",
        "model": "mlp-128",
        "metric": "macro_f1",
        "transform": "identity",
        "flops": 4.0e6,
        "options": {
            "batch_size": 256,
            "lcpfn_alignment": "LCBench modality and MLP family",
            "difficulty_tier": "hard",
        },
        "datasets": [
            {"id": "covertype", "key": "covertype", "name": "Forest Covertype (7 classes)"},
            {"id": "sensorless-drive", "key": "sensorless_drive", "name": "Sensorless Drive Diagnosis (11 classes)"},
            {"id": "letter-recognition", "key": "letter_recognition", "name": "Letter Recognition (26 classes)"},
        ],
    },
    {
        "id": "tabular_regression",
        "title": "Hard Tabular Regression",
        "task_type": "tabular_regression",
        "task_family": "tabular",
        "case_suffix": "tabular-regression",
        "model": "mlp-regressor-128",
        "metric": "nrmse",
        "transform": "inverse_positive",
        "flops": 4.0e6,
        "options": {
            "batch_size": 512,
            "max_tabular_samples": 200000,
            "standardize_regression_target": True,
            "difficulty_tier": "hard",
        },
        "datasets": [
            {"id": "yearprediction-msd", "key": "year_prediction_msd", "name": "YearPredictionMSD"},
            {"id": "online-news-popularity", "key": "online_news_popularity", "name": "Online News Popularity"},
            {"id": "superconductivity", "key": "superconductivity", "name": "Superconductivity Critical Temperature"},
        ],
    },
    {
        "id": "anomaly_detection",
        "title": "Hard Anomaly Detection",
        "task_type": "anomaly_detection",
        "task_family": "tabular",
        "case_suffix": "anomaly-detection",
        "model": "autoencoder-128",
        "metric": "auroc",
        "transform": "identity",
        "flops": 6.0e6,
        "options": {"batch_size": 512, "max_tabular_samples": 200000, "difficulty_tier": "hard"},
        "datasets": [
            {"id": "kddcup99-full", "key": "kddcup99", "name": "KDD Cup 1999 (full 10% mixture)", "options": {"kdd_subset": "full"}},
            {"id": "covertype-minority", "key": "covertype_anomaly", "name": "Forest Covertype Minority Anomaly"},
            {"id": "sensorless-minority", "key": "sensorless_drive_anomaly", "name": "Sensorless Drive Minority Anomaly"},
        ],
    },
    {
        "id": "text_classification",
        "title": "Hard Text Classification",
        "task_type": "text_classification",
        "task_family": "text_classification",
        "case_suffix": "text-classification",
        "model": "bidirectional-gru-hash30k",
        "metric": "macro_f1",
        "transform": "identity",
        "flops": 1.2e9,
        "options": {
            "batch_size": 64,
            "max_train_samples": 50000,
            "validation_samples": 5000,
            "max_test_samples": 10000,
            "sequence_length": 384,
            "vocabulary_size": 30000,
            "difficulty_tier": "hard",
        },
        "datasets": [
            {"id": "imdb", "key": "imdb", "name": "IMDB Reviews", "options": {"lcpfn_alignment": "Taskset IMDB sentiment and RNN family"}},
            {"id": "ag-news", "key": "ag_news", "name": "AG News (4 classes)", "options": {"num_classes": 4}},
            {"id": "yelp-review-full", "key": "yelp_review_full", "name": "Yelp Review Full (5 classes)", "options": {"num_classes": 5}},
        ],
    },
    {
        "id": "language_modeling",
        "title": "Hard Language Modeling",
        "task_type": "language_modeling",
        "task_family": "language_modeling",
        "case_suffix": "language-modeling",
        "model": "transformer-encoder-lm-hash16k",
        "metric": "cross_entropy",
        "transform": "exp_negative",
        "flops": 4.0e9,
        "options": {
            "batch_size": 32,
            "sequence_length": 96,
            "vocabulary_size": 16000,
            "max_train_tokens": 600000,
            "difficulty_tier": "hard",
        },
        "datasets": [
            {"id": "wikitext103", "key": "wikitext103", "name": "WikiText-103", "options": {"lcpfn_alignment": "PD1 LM1B Transformer proxy"}},
            {"id": "wikitext2", "key": "wikitext2", "name": "WikiText-2"},
            {"id": "penn-treebank", "key": "penn_treebank", "name": "Penn Treebank"},
        ],
    },
    {
        "id": "time_series_forecasting",
        "title": "Hard Multi-Step Time-Series Forecasting",
        "task_type": "time_series_forecasting",
        "task_family": "time_series_forecasting",
        "case_suffix": "time-series-forecasting",
        "model": "lstm-multistep-forecaster",
        "metric": "nrmse",
        "transform": "inverse_positive",
        "flops": 8.0e8,
        "options": {"batch_size": 128, "sequence_length": 192, "difficulty_tier": "hard"},
        "datasets": [
            {"id": "etth1-h24", "key": "etth1", "name": "ETTh1 24-step", "options": {"forecast_horizon": 24}},
            {"id": "ettm1-h48", "key": "ettm1", "name": "ETTm1 48-step", "options": {"forecast_horizon": 48}},
            {"id": "ettm2-h96", "key": "ettm2", "name": "ETTm2 96-step", "options": {"forecast_horizon": 96}},
        ],
    },
    {
        "id": "recommendation",
        "title": "Hard Recommendation",
        "task_type": "recommendation",
        "task_family": "recommendation",
        "case_suffix": "recommendation",
        "model": "large-hashed-matrix-factorization",
        "metric": "rmse",
        "transform": "inverse_positive",
        "flops": 5.0e7,
        "options": {
            "batch_size": 2048,
            "user_buckets": 131072,
            "item_buckets": 32768,
            "max_samples": 1000000,
            "difficulty_tier": "hard",
        },
        "datasets": [
            {"id": "movielens1m", "key": "movielens1m", "name": "MovieLens 1M"},
            {"id": "movielens10m", "key": "movielens10m", "name": "MovieLens 10M"},
            {"id": "movielens20m", "key": "movielens20m", "name": "MovieLens 20M"},
        ],
    },
    {
        "id": "reinforcement_learning",
        "title": "Hard Reinforcement Learning",
        "task_type": "reinforcement_learning",
        "task_family": "reinforcement_learning",
        "case_suffix": "reinforcement-learning",
        "model": "padded-state-dqn",
        "metric": "normalized_eval_reward",
        "transform": "identity",
        "flops": 2.0e6,
        "options": {
            "progress_unit": "evaluation_cycle",
            "episodes_per_checkpoint": 10,
            "evaluation_episodes": 20,
            "difficulty_tier": "hard",
        },
        "datasets": [
            {"id": "cartpole-heavy", "key": "cartpole_heavy", "name": "CartPole Heavy Pole"},
            {"id": "cartpole-noisy", "key": "cartpole_noisy", "name": "CartPole Stochastic Dynamics"},
            {"id": "mountaincar", "key": "mountaincar", "name": "MountainCar Sparse Reward"},
        ],
    },
]


def _cross_domain_stages(
    definitions: list[dict[str, Any]],
    *,
    checkpoint_root: str,
) -> list[dict[str, Any]]:
    return [
        {
            "id": definition["id"],
            "title": definition["title"],
            "enabled": True,
            "purpose": "Three-dataset scratch-versus-source-domain-pretrained generalisation pair.",
            "cases": _paired_cases(definition, checkpoint_root=checkpoint_root),
        }
        for definition in definitions
    ]


CROSS_DOMAIN_STAGES = _cross_domain_stages(
    TASK_DEFINITIONS,
    checkpoint_root="/workspace-cache/controller-pretrained-cross-domain",
)

HARD_CROSS_DOMAIN_STAGES = _cross_domain_stages(
    HARD_TASK_DEFINITIONS,
    checkpoint_root="/workspace-cache/controller-pretrained-cross-domain-hard-v1",
)

EXPENSIVE_TASK_DEFINITIONS = [
    {
        "id": "text_classification",
        "title": "Expensive Transformer Text Classification",
        "task_type": "text_classification",
        "task_family": "text_classification",
        "case_suffix": "text-classification",
        "model": "transformer-text-30k",
        "metric": "macro_f1",
        "transform": "identity",
        "flops": 8.0e9,
        "options": {
            "batch_size": 48,
            "max_train_samples": 60000,
            "validation_samples": 5000,
            "max_test_samples": 10000,
            "sequence_length": 384,
            "vocabulary_size": 30000,
            "model_width": 256,
            "model_layers": 4,
            "attention_heads": 8,
            "learning_rate": 0.0005,
            "weight_decay": 0.01,
            "difficulty_tier": "expensive",
            "require_cuda": True,
            "lcpfn_alignment": "Taskset-style text classification with a transformer sequence model",
        },
        "datasets": [
            {"id": "ag-news", "key": "ag_news", "name": "AG News Transformer (4 classes)", "options": {"num_classes": 4}},
            {"id": "imdb", "key": "imdb", "name": "IMDB Transformer Sentiment"},
            {"id": "yelp-review-full", "key": "yelp_review_full", "name": "Yelp Review Full Transformer (5 classes)", "options": {"num_classes": 5}},
        ],
    },
    {
        "id": "language_modeling",
        "title": "Expensive Transformer Language Modeling",
        "task_type": "language_modeling",
        "task_family": "language_modeling",
        "case_suffix": "language-modeling",
        "model": "transformer-lm-expensive-24k",
        "metric": "cross_entropy",
        "transform": "exp_negative",
        "flops": 1.2e10,
        "options": {
            "batch_size": 24,
            "sequence_length": 128,
            "vocabulary_size": 24000,
            "max_train_tokens": 1200000,
            "model_width": 256,
            "model_layers": 5,
            "attention_heads": 8,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "difficulty_tier": "expensive",
            "require_cuda": True,
            "lcpfn_alignment": "PD1/LM1B-style transformer language-model learning curve proxy",
        },
        "datasets": [
            {"id": "wikitext103", "key": "wikitext103", "name": "WikiText-103 Transformer LM"},
            {"id": "wikitext2", "key": "wikitext2", "name": "WikiText-2 Transformer LM"},
            {"id": "penn-treebank", "key": "penn_treebank", "name": "Penn Treebank Transformer LM"},
        ],
    },
    {
        "id": "time_series_forecasting",
        "title": "Expensive Transformer Time-Series Forecasting",
        "task_type": "time_series_forecasting",
        "task_family": "time_series_forecasting",
        "case_suffix": "time-series-forecasting",
        "model": "timeseries-transformer",
        "metric": "nrmse",
        "transform": "inverse_positive",
        "flops": 6.0e9,
        "options": {
            "batch_size": 96,
            "sequence_length": 256,
            "model_width": 256,
            "model_layers": 4,
            "attention_heads": 8,
            "learning_rate": 0.0007,
            "weight_decay": 0.001,
            "difficulty_tier": "expensive",
            "require_cuda": True,
            "lcpfn_alignment": "Long sequence forecasting with transformer encoder",
        },
        "datasets": [
            {"id": "etth1-h48", "key": "etth1", "name": "ETTh1 48-step Transformer", "options": {"forecast_horizon": 48}},
            {"id": "ettm1-h96", "key": "ettm1", "name": "ETTm1 96-step Transformer", "options": {"forecast_horizon": 96}},
            {"id": "ettm2-h192", "key": "ettm2", "name": "ETTm2 192-step Transformer", "options": {"forecast_horizon": 192}},
        ],
    },
    {
        "id": "recommendation",
        "title": "Expensive Neural Recommendation",
        "task_type": "recommendation",
        "task_family": "recommendation",
        "case_suffix": "recommendation",
        "model": "neural-cf-large",
        "metric": "rmse",
        "transform": "inverse_positive",
        "flops": 2.0e9,
        "options": {
            "batch_size": 4096,
            "user_buckets": 262144,
            "item_buckets": 65536,
            "embedding_dim": 128,
            "model_width": 768,
            "max_samples": 2000000,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "difficulty_tier": "expensive",
            "require_cuda": True,
        },
        "datasets": [
            {"id": "movielens1m", "key": "movielens1m", "name": "MovieLens 1M Neural CF"},
            {"id": "movielens10m", "key": "movielens10m", "name": "MovieLens 10M Neural CF"},
            {"id": "movielens20m", "key": "movielens20m", "name": "MovieLens 20M Neural CF"},
        ],
    },
    {
        "id": "reinforcement_learning",
        "title": "Expensive Reinforcement Learning",
        "task_type": "reinforcement_learning",
        "task_family": "reinforcement_learning",
        "case_suffix": "reinforcement-learning",
        "model": "padded-state-dqn-expensive",
        "metric": "normalized_eval_reward",
        "transform": "identity",
        "flops": 5.0e7,
        "options": {
            "progress_unit": "evaluation_cycle",
            "episodes_per_checkpoint": 25,
            "evaluation_episodes": 30,
            "difficulty_tier": "expensive",
            "require_cuda": True,
        },
        "datasets": [
            {"id": "cartpole-heavy", "key": "cartpole_heavy", "name": "CartPole Heavy Pole DQN"},
            {"id": "cartpole-noisy", "key": "cartpole_noisy", "name": "CartPole Stochastic DQN"},
            {"id": "mountaincar", "key": "mountaincar", "name": "MountainCar Sparse Reward DQN"},
        ],
    },
]

EXPENSIVE_CROSS_DOMAIN_STAGES = _cross_domain_stages(
    EXPENSIVE_TASK_DEFINITIONS,
    checkpoint_root="/workspace-cache/controller-pretrained-cross-domain-expensive-v1",
)


HARDEST_TASK_DEFINITIONS = [
    {
        "id": "text_classification",
        "title": "Frontier Multi-Label and Multilingual Text Classification",
        "task_type": "text_classification",
        "task_family": "text_classification",
        "case_suffix": "text-classification",
        "model": "transformer-text-frontier-40k",
        "metric": "macro_f1",
        "transform": "identity",
        "flops": 2.4e10,
        "options": {
            "batch_size": 48,
            "max_train_samples": 60000,
            "validation_samples": 5000,
            "max_test_samples": 5000,
            "sequence_length": 512,
            "vocabulary_size": 40000,
            "model_width": 384,
            "model_layers": 6,
            "attention_heads": 8,
            "learning_rate": 0.00035,
            "weight_decay": 0.01,
            "difficulty_tier": "hardest_practical",
            "require_cuda": True,
        },
        "datasets": [
            {"id": "go-emotions", "key": "go_emotions", "name": "GoEmotions (28-label multi-label)", "options": {"multilabel": True, "num_classes": 28}},
            {"id": "multi-eurlex", "key": "multi_eurlex", "name": "MultiEURLEX (long legal multi-label)", "options": {"multilabel": True, "num_classes": 21}},
            {"id": "marc-multilingual", "key": "marc_multilingual", "name": "MARC Multilingual Reviews (6 languages, 5 classes)", "options": {"num_classes": 5}},
        ],
    },
    {
        "id": "language_modeling",
        "title": "Frontier Transformer Language Modeling",
        "task_type": "language_modeling",
        "task_family": "language_modeling",
        "case_suffix": "language-modeling",
        "model": "transformer-lm-frontier-32k",
        "metric": "cross_entropy",
        "transform": "exp_negative",
        "flops": 3.6e10,
        "options": {
            "batch_size": 16,
            "sequence_length": 256,
            "vocabulary_size": 32000,
            "max_train_tokens": 2000000,
            "model_width": 384,
            "model_layers": 6,
            "attention_heads": 8,
            "learning_rate": 0.00025,
            "weight_decay": 0.01,
            "difficulty_tier": "hardest_practical",
            "require_cuda": True,
            "lcpfn_alignment": "Includes the PD1/LC-PFN LM1B task family with an online single-run protocol",
        },
        "datasets": [
            {"id": "wikitext103", "key": "wikitext103", "name": "WikiText-103"},
            {"id": "lm1b-shard", "key": "lm1b_shard", "name": "One Billion Word fixed shard"},
            {"id": "c4-en-shard", "key": "c4_en_shard", "name": "C4 English fixed shard"},
        ],
    },
    {
        "id": "time_series_forecasting",
        "title": "Frontier Long-Horizon Multi-Series Forecasting",
        "task_type": "time_series_forecasting",
        "task_family": "time_series_forecasting",
        "case_suffix": "time-series-forecasting",
        "model": "timeseries-transformer-frontier",
        "metric": "nrmse",
        "transform": "inverse_positive",
        "flops": 1.8e10,
        "options": {
            "batch_size": 64,
            "sequence_length": 336,
            "forecast_horizon": 720,
            "max_train_windows": 20000,
            "max_validation_windows": 4000,
            "max_test_windows": 4000,
            "model_width": 256,
            "model_layers": 4,
            "attention_heads": 8,
            "learning_rate": 0.0005,
            "weight_decay": 0.001,
            "difficulty_tier": "hardest_practical",
            "require_cuda": True,
        },
        "datasets": [
            {"id": "traffic-hourly", "key": "monash_traffic_hourly", "name": "Monash Traffic Hourly (862 series)"},
            {"id": "electricity-hourly", "key": "monash_electricity_hourly", "name": "Monash Electricity Hourly (321 series)"},
            {"id": "solar-10min", "key": "monash_solar_10_minutes", "name": "Monash Solar 10-Minute (137 series)"},
        ],
    },
    {
        "id": "recommendation",
        "title": "Frontier Sparse Neural Recommendation",
        "task_type": "recommendation",
        "task_family": "recommendation",
        "case_suffix": "recommendation",
        "model": "neural-cf-frontier",
        "metric": "ndcg_at_10",
        "transform": "identity",
        "flops": 4.0e9,
        "options": {
            "batch_size": 4096,
            "user_buckets": 524288,
            "item_buckets": 131072,
            "embedding_dim": 128,
            "model_width": 768,
            "max_samples": 1500000,
            "max_evaluation_interactions": 1000,
            "learning_rate": 0.0008,
            "weight_decay": 0.0001,
            "difficulty_tier": "hardest_practical",
            "require_cuda": True,
        },
        "datasets": [
            {"id": "movielens25m", "key": "movielens25m", "name": "MovieLens 25M"},
            {"id": "amazon-books-5core", "key": "amazon_books_5core", "name": "Amazon Reviews 2023 Books 5-core"},
            {"id": "amazon-electronics-5core", "key": "amazon_electronics_5core", "name": "Amazon Reviews 2023 Electronics 5-core"},
        ],
    },
    {
        "id": "reinforcement_learning",
        "title": "Hard Learnable MinAtar Reinforcement Learning",
        "task_type": "reinforcement_learning",
        "task_family": "reinforcement_learning",
        "case_suffix": "reinforcement-learning",
        "model": "minatar-cnn-dqn-frontier",
        "metric": "normalized_eval_reward",
        "transform": "identity",
        "flops": 1.5e9,
        "options": {
            "progress_unit": "evaluation_cycle",
            "episodes_per_checkpoint": 10,
            "evaluation_episodes": 10,
            "maximum_episode_steps": 2000,
            "difficulty_tier": "hardest_practical",
            "require_cuda": True,
        },
        "datasets": [
            {"id": "minatar-breakout", "key": "minatar_breakout", "name": "MinAtar Breakout", "options": {"reward_scale": 10.0}},
            {"id": "minatar-seaquest", "key": "minatar_seaquest", "name": "MinAtar Seaquest", "options": {"reward_scale": 20.0}},
            {"id": "minatar-asterix", "key": "minatar_asterix", "name": "MinAtar Asterix", "options": {"reward_scale": 20.0}},
        ],
    },
]

HARDEST_CROSS_DOMAIN_STAGES = _cross_domain_stages(
    HARDEST_TASK_DEFINITIONS,
    checkpoint_root="/workspace-cache/controller-pretrained-cross-domain-hardest-v1",
)


def build_manifest(
    vision_source: dict[str, Any],
    classification_source: dict[str, Any],
    pretrained_source: dict[str, Any] | None = None,
    *,
    include_vision: bool = False,
    suite: str = "standard",
) -> dict[str, Any]:
    if suite not in {"standard", "hard", "expensive", "hardest"}:
        raise ValueError(f"Unsupported generalized benchmark suite: {suite}")
    manifest = copy.deepcopy(vision_source)
    classification_stage = copy.deepcopy(classification_source["stages"][0])
    manifest["stages"] = (
        [
            classification_stage if stage["id"] == "image_classification" else stage
            for stage in manifest["stages"]
        ]
        if include_vision
        else []
    )
    for stage in manifest["stages"]:
        for case in stage["cases"]:
            case.setdefault("quality_definition", {"metric": case.get("primary_metric", "quality_score"), "transform": "identity", "scale": 1.0})
            case.setdefault("progress_unit", "epoch")
            if case.get("dataset_key") in {"cifar10", "cifar100"}:
                case.setdefault("lcpfn_alignment", "NAS-Bench-201 dataset; different architecture search protocol")
    if include_vision and pretrained_source is not None:
        for source_stage in pretrained_source["stages"]:
            stage = copy.deepcopy(source_stage)
            stage["id"] = f"{stage['id']}_transfer_learning"
            stage["title"] = f"Transfer Learning: {stage['title']}"
            # Some source manifests contain paired scratch and pretrained
            # segmentation cases. The transfer-learning half must include only
            # pretrained cases; otherwise the combined thesis matrix contains
            # three duplicated scratch cases (51 instead of 48).
            stage["cases"] = [
                case for case in stage["cases"] if bool(case.get("pretrained", False))
            ]
            for case in stage["cases"]:
                case.setdefault(
                    "quality_definition",
                    {
                        "metric": case.get("primary_metric", "quality_score"),
                        "transform": "identity",
                        "scale": 1.0,
                    },
                )
                case.setdefault("progress_unit", "epoch")
                if case.get("dataset_key") in {"cifar10", "cifar100"}:
                    case.setdefault(
                        "lcpfn_alignment",
                        "NAS-Bench-201 dataset with transfer initialization; different training protocol",
                    )
            manifest["stages"].append(stage)
    stages = (
        HARDEST_CROSS_DOMAIN_STAGES
        if suite == "hardest"
        else EXPENSIVE_CROSS_DOMAIN_STAGES
        if suite == "expensive"
        else HARD_CROSS_DOMAIN_STAGES if suite == "hard" else CROSS_DOMAIN_STAGES
    )
    benchmark_version = (
        "rapec-generalized-nonvision-hardest-v1"
        if suite == "hardest"
        else "rapec-generalized-nonvision-expensive-v1"
        if suite == "expensive"
        else "rapec-generalized-nonvision-hard-v1"
        if suite == "hard"
        else "rapec-generalized-nonvision-v2"
    )
    manifest["stages"].extend(copy.deepcopy(stages))
    manifest["benchmark_version"] = benchmark_version
    manifest["description"] = (
        "Hardest practical GPU-required suite with multi-label text, LM1B/C4, 720-step forecasting, "
        "sparse ranking recommendation, and license-free hard learnable MinAtar control."
        if suite == "hardest"
        else "Expensive GPU-required non-vision benchmark with transformer, neural recommendation, "
        "and longer control runs for YOLO-like epoch cost and quality dynamics."
        if suite == "expensive"
        else "Hard live-shadow generalisation benchmark with larger or noisier datasets, multi-step "
        "forecasting, stochastic control, and paired scratch/transfer initialization."
        if suite == "hard"
        else "Live-shadow generalisation benchmark with three datasets and paired scratch/transfer "
        "initialization for each non-vision task family. Existing vision suites remain separate."
    )
    manifest["execution"].update(
        {
            "experiment_name": benchmark_version,
            "timeout_seconds": 86400 if suite == "hardest" else 129600 if suite == "expensive" else 86400 if suite == "hard" else 43200,
            "analysis_module": "controller_benchmark.analyze_live_shadow",
            "dashboard_files": ["rapec-generalized-cross-domain.json"],
        }
    )
    manifest["acceptance"] = {
        "minimum_energy_saving_fraction": 0.0,
        "maximum_quality_regret": 0.10,
        "minimum_success_rate": 0.80,
    }
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-source", type=Path, required=True)
    parser.add_argument("--classification-source", type=Path, required=True)
    parser.add_argument("--pretrained-source", type=Path)
    parser.add_argument("--include-vision", action="store_true")
    parser.add_argument("--suite", choices=("standard", "hard", "expensive", "hardest"), default="standard")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vision = json.loads(args.vision_source.read_text(encoding="utf-8"))
    classification = json.loads(args.classification_source.read_text(encoding="utf-8"))
    pretrained = (
        json.loads(args.pretrained_source.read_text(encoding="utf-8"))
        if args.pretrained_source
        else None
    )
    manifest = build_manifest(
        vision,
        classification,
        pretrained,
        include_vision=args.include_vision,
        suite=args.suite,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({"stages": len(manifest["stages"]), "cases": sum(len(stage["cases"]) for stage in manifest["stages"]), "task_types": sorted({case["task_type"] for stage in manifest["stages"] for case in stage["cases"]})}, indent=2))


if __name__ == "__main__":
    main()
