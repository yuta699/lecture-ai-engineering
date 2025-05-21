import os
import pytest
import pandas as pd
import numpy as np
import great_expectations as gx
from sklearn.datasets import fetch_openml
import warnings

# 警告を抑制
warnings.filterwarnings("ignore")

# テスト用データパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")


@pytest.fixture
def sample_data():
    """Titanicテスト用データセットを読み込む"""
    return pd.read_csv(DATA_PATH)


def test_data_exists(sample_data):
    """データが存在することを確認"""
    assert not sample_data.empty, "データセットが空です"
    assert len(sample_data) > 0, "データセットにレコードがありません"


def test_data_columns(sample_data):
    """必要なカラムが存在することを確認"""
    expected_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Survived",
    ]
    for col in expected_columns:
        assert (
            col in sample_data.columns
        ), f"カラム '{col}' がデータセットに存在しません"


def test_data_types(sample_data):
    """データ型の検証"""
    # 数値型カラム
    numeric_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    for col in numeric_columns:
        assert pd.api.types.is_numeric_dtype(
            sample_data[col].dropna()
        ), f"カラム '{col}' が数値型ではありません"

    # カテゴリカルカラム
    categorical_columns = ["Sex", "Embarked"]
    for col in categorical_columns:
        assert (
            sample_data[col].dtype == "object"
        ), f"カラム '{col}' がカテゴリカル型ではありません"

    # 目的変数
    survived_vals = sample_data["Survived"].dropna().unique()
    assert set(survived_vals).issubset({"0", "1"}) or set(survived_vals).issubset(
        {0, 1}
    ), "Survivedカラムには0, 1のみ含まれるべきです"


def test_missing_values_acceptable(sample_data):
    """欠損値の許容範囲を確認"""
    # 完全に欠損するのではなく、許容範囲内の欠損を確認
    for col in sample_data.columns:
        missing_rate = sample_data[col].isna().mean()
        assert (
            missing_rate < 0.8
        ), f"カラム '{col}' の欠損率が80%を超えています: {missing_rate:.2%}"


def test_value_ranges(sample_data):
    """値の範囲を手動で検証"""

    required_columns = ["Pclass", "Sex", "Age", "Fare", "Embarked"]

    # 必要カラムのチェック
    missing_columns = [col for col in required_columns if col not in sample_data.columns]
    assert not missing_columns, f"警告: 以下のカラムがありません: {missing_columns}"

    # Pclass は 1, 2, 3 のみ
    assert sample_data["Pclass"].dropna().isin([1, 2, 3]).all(), "Pclassに不正な値があります"

    # Sex は male / female のみ
    assert sample_data["Sex"].dropna().isin(["male", "female"]).all(), "Sexに不正な値があります"

    # Age は 0～100 の範囲
    assert sample_data["Age"].dropna().between(0, 100).all(), "Ageに不正な値があります"

    # Fare は 0～600 の範囲
    assert sample_data["Fare"].dropna().between(0, 600).all(), "Fareに不正な値があります"

    # Embarked は C / Q / S のいずれか
    assert sample_data["Embarked"].dropna().isin(["C", "Q", "S"]).all(), "Embarkedに不正な値があります"

# Great Expectations部分はひとまずコメントアウト




   
