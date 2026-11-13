"""Giai đoạn 1b: chọn bộ phân loại và các giá trị tham số cần thử.

File này CHƯA huấn luyện. experiments.py ghép bộ phân loại với bộ tạo vector,
sau đó gọi fit trên từng fold train và chọn tham số bằng CV.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


MODELS = ("Naive Bayes", "SVM")
NB_ALPHAS = [0.1, 0.3, 1.0]
SVM_CS = [0.1, 0.3, 1.0, 3.0]


def make_classifier(model_name, *, seed=42, class_weight=None):
    """Trả (bộ phân loại chưa fit, các giá trị tham số để CV thử).

    alpha: cộng thêm vào số đếm NB để một từ chưa gặp không có xác suất 0.
    C: mức phạt lỗi khi SVM học ranh giới; cần thử trên validation.
    Ba giá trị alpha và bốn giá trị C là lưới nhỏ do nhóm chọn để khảo sát,
    không phải kết quả đã học. class_weight mặc định None; S1 dùng balanced.
    """
    if model_name == "Naive Bayes":
        # NB nhận vector không âm; chưa biết nhãn ham/spam cho đến lúc fit.
        if class_weight is not None:
            raise ValueError("class_weight chỉ dùng cho SVM trong thí nghiệm này")
        classifier = MultinomialNB()
        grid = {"clf__alpha": NB_ALPHAS}
    elif model_name == "SVM":
        # dual=False giữ cách giải bài toán gốc của project; max_iter=20000 là
        # số vòng tối đa để bộ tối ưu hội tụ. random_state được giữ trong cấu
        # hình, nhưng scikit-learn không dùng nó khi dual=False.
        # Chưa có trọng số phân loại nào được học ở dòng này.
        classifier = LinearSVC(dual=False, max_iter=20000, random_state=seed)
        grid = {"clf__C": SVM_CS,
                "clf__class_weight": [class_weight]}
    else:
        raise ValueError(f"Bộ phân loại không hỗ trợ: {model_name}")
    return classifier, grid
