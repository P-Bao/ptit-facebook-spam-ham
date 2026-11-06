import numpy as np
import pandas as pd
import pytest

from pipeline.modern.data import class_weights, normalize_text, prepare_data
from pipeline.modeling.training_utils import bucket, norm_key


def test_modern_normalization_preserves_case_stopwords_and_negation():
    assert normalize_text("  PTIT\nKhông &amp; có  ") == "PTIT Không & có"


def test_modern_split_keeps_existing_outer_test_and_groups(tmp_path):
    frame = pd.DataFrame([{"id":str(i),"text":f"Bình luận mã{chr(97+(i//2)//26)}{chr(97+(i//2)%26)}!",
                           "cleaned_text":f"Nội dung mã{chr(97+(i//2)//26)}{chr(97+(i//2)%26)}",
                           "label":"spam" if (i//2)%2 else "ham"} for i in range(240)])
    path=tmp_path / "data.csv"
    frame.to_csv(path,index=False)
    result=prepare_data(path)
    expected=[bucket(norm_key(text),"test")==0 for text in frame.cleaned_text]
    assert result.outer_split.eq("test").tolist()==expected
    assert result.groupby("cleaned_text").development_split.nunique().max()==1
    pd.testing.assert_frame_equal(result,prepare_data(path))


def test_modern_class_weights_use_only_supplied_labels():
    np.testing.assert_allclose(class_weights([0,0,0,1]),[2/3,2])
    with pytest.raises(ValueError):class_weights([0,0])
