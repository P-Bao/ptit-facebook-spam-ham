import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from pipeline.step_02_preprocess.text_preprocessing import (
    EMOJI, clean_text, mask_names, preprocess,
)
from pipeline.step_05_features.spam_features import FastTextVectorizer
from pipeline.step_06_train_evaluate.training_utils import (
    grouped_folds, norm_key, sample_training_indices, undersampled_search,
)


@pytest.mark.parametrize("text", ["Máy Tính học tốt", "Mạng Máy Tính khó quá", "Người tham gia ẩn danh 123"])
def test_non_names_are_not_masked(text):
    assert mask_names(text) == text


def test_name_and_username_are_masked_conservatively():
    assert mask_names("Nguyễn Văn An ơi") == "user_name ơi"
    assert mask_names("HappyOwl1234 xin chào").startswith("user_name")
    # Chỉ hứa che cụm đầu câu, không hứa ẩn danh hoàn toàn.
    assert mask_names("bạn Nguyễn Văn An ơi") == "bạn Nguyễn Văn An ơi"


def test_raw_and_cleaned_inference_have_same_tokens():
    text = "Nguyễn Văn An xem https://example.com gọi 090.123.4567 nhé 😂😂"
    result = preprocess(text, text)
    assert result == preprocess(clean_text(text), text)
    assert "urltoken" in result and "phonetoken" in result
    assert result.split().count("emoji_token") == 1
    assert not EMOJI.search(result)


def test_negation_and_digits_survive():
    result = preprocess("không mua 100000 đồng", remove_stop=True)
    assert "không" in result and "100000" in result


def test_punctuation_group_is_not_row_dependent():
    assert norm_key(".") == norm_key(" . ")
    assert norm_key("Mã 123!") == norm_key("mã 456")


def test_grouped_cv_keeps_duplicates_together():
    keys = np.array([f"text{i}" for i in range(80) for _ in range(2)])
    y = np.array([i % 2 for i in range(80) for _ in range(2)])
    folds = grouped_folds(keys, y)
    coverage = np.zeros(len(y), dtype=int)
    for train, valid in folds:
        assert not set(keys[train]) & set(keys[valid])
        coverage[valid] += 1
    assert (coverage == 1).all()


def test_sampling_only_uses_allowed_training_indices():
    y = np.array([1] * 10 + [0] * 60)
    allowed = np.arange(5, 65)
    chosen = sample_training_indices(y, allowed)
    assert set(chosen).issubset(allowed)
    assert (y[chosen] == 0).sum() == 2 * (y[chosen] == 1).sum()


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    fits = []
    predictions = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.__class__.fits.append(set(X.tolist()))
        return self

    def predict(self, X):
        self.__class__.predictions.append(set(X.tolist()))
        return (X % 2).astype(int)


def test_undersampling_leaves_validation_untouched():
    X = np.arange(60)
    y = X % 2
    folds = [(np.arange(20, 60), np.arange(20)),
             (np.r_[0:20, 40:60], np.arange(20, 40)),
             (np.arange(40), np.arange(40, 60))]
    RecordingClassifier.fits = []
    RecordingClassifier.predictions = []
    undersampled_search(RecordingClassifier(), {}, X, y, folds)
    for i, (train, valid) in enumerate(folds):
        assert RecordingClassifier.fits[i].issubset(set(train))
        assert not RecordingClassifier.fits[i] & set(valid)
        assert RecordingClassifier.predictions[i] == set(valid)


def test_fasttext_oov_and_empty_sentence():
    vec = FastTextVectorizer(vector_size=8, min_count=1, epochs=2, bucket=2000, workers=1)
    vec.fit(["xin chào bạn", "học môn mới", "xin tài liệu", "chào bạn học"])
    transformed = vec.transform(["inboxxx", ""])
    assert transformed.shape == (2, 8)
    assert np.isfinite(transformed).all()
    assert np.all(transformed[1] == 0)
