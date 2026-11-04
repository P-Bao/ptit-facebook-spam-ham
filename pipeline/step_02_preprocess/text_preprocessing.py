"""Tiền xử lý dùng chung cho train và dự đoán; notebook nhúng nguyên module này."""

import html
import re
import unicodedata

from underthesea import word_tokenize

# Danh sách cố định kế thừa từ notebook nhóm; không tải từ điển qua mạng khi chạy.
# Nguồn được ghi ở notebook gốc: hoten.org. Chỉ dùng để nhận diện bảo thủ ở đầu câu.
SURNAMES_TEXT = """
Nguyễn Trần Lê Phạm Huỳnh Hoàng Võ Vũ Phan Trương Bùi Đặng Đỗ Ngô Hồ Dương Đinh Đoàn
Lâm Mai Trịnh Đào Cao Lý Hà Lưu Lương Thái Châu Tạ Phùng Tô Vương Văn Tăng Quách Lại
Hứa Thạch Diệp Từ Chu La Đàm Tống Giang Chung Triệu Kiều Hồng Trang Đồng Danh Lư Lữ
Thân Kim Mã Bạch Liêu Tiêu Dư Bành Âu Tôn Khưu Sơn Tất Nghiêm Lục Quan Phương Mạc
Lai Vòng Mạch Thiều Trà Đậu Nhan Lã Trình Ninh Trầm Vi Biện Hàng Ôn Chế Nhâm Thi
Doãn Khổng Phù Đường Ông Ngụy Viên Tào Cù
"""
GIVEN_NAMES_TEXT = """
Anh Vy Ngọc Nhi Hân Thư Linh Như Ngân Phương Thảo My Trân Quỳnh Nghi Trang Trâm An Thy
Châu Trúc Uyên Yến Ý Tiên Mai Hà Vân Nguyên Hương Quyên Duyên Kim Trinh Thanh Tuyền Hằng
Dương Chi Giang Tâm Lam Tú Ánh Hiền Khánh Minh Huyền Thùy Vi Ly Dung Nhung Phúc Lan Phụng
Ân Thi Khanh Kỳ Nga Tường Thúy Mỹ Hoa Tuyết Lâm Thủy Đan Hạnh Xuân Oanh Mẫn Khuê Diệp
Thương Nhiên Băng Hồng Bình Loan Thơ Phượng Mi Nhã Nguyệt Bích Đào Diễm Kiều Hiếu Di Liên
Trà Tuệ Thắm Diệu Quân Nhàn Doanh Huy Khang Bảo Khoa Phát Đạt Khôi Long Nam Duy Kiệt
Thịnh Tuấn Hưng Hoàng Nhân Trí Tài Phong Phú Thành Đức Dũng Lộc Vinh Tiến Nghĩa Thiện Hào
Hải Đăng Quang Nhật Trung Thắng Hùng Sang Sơn Thái Cường Vũ Toàn Thuận Trường Danh Kiên
Phước Thiên Tân Việt Khải Tín Tùng Quý Hậu Trọng Triết Luân Quốc Thông Khiêm Hòa Kha Vỹ
Bách Mạnh Lợi Đại Hiệp Đông Nhựt Phi Tấn Văn Vương Công Hiển Vĩ Nữ Thất
"""


def strip_accents(text):
    text = unicodedata.normalize("NFD", str(text).replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


SURNAMES = {strip_accents(w) for w in SURNAMES_TEXT.split()}
NAME_WORDS = SURNAMES | {strip_accents(w) for w in GIVEN_NAMES_TEXT.split()}
STOP_WORDS = set("""
và của là có được cho với các những một này đó ấy kia thì mà ở tại trong ngoài trên dưới
khi nếu nhưng hay hoặc vì nên để bởi do từ đến tới về theo như cũng đã đang sẽ vẫn còn
rất quá lắm hơn nhất rồi xong đều cả mọi mỗi nào gì sao đâu ai thế vậy đây nè nhé nha
nhỉ ạ à ừ ờ ơi hả hử chứ thôi luôn thật lại ra vào lên xuống đi qua nữa chỉ mới cùng bị
bởi_vì cho_nên vì_vậy do_đó tuy_nhiên mặc_dù như_vậy thế_nào bao_giờ tất_cả cái những_gì
""".split())
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+84|0084|84|0)(?:[\s.-]?[3-9](?:[\s.-]?\d){8})(?!\d)")
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2300-\u23FF"
                   "\u2190-\u21FF\u2900-\u297F\u3030\u303D\u3297\u3299\uFE0F\u200d\u20E3\u00A9\u00AE\u2122]")
USERNAME = re.compile(r"\b(?:(?:[A-Z][a-z]+){2,}\d{2,4}|[A-Za-z_]*[A-Za-z]\w*\d{3,})\b")
REPEAT_LETTER = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)


def clean_text(text):
    """Giữ dấu hiệu URL/SĐT; không loại câu ngắn hoặc tự đổi nhãn."""
    text = unicodedata.normalize("NFC", html.unescape(str(text)))
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL.sub(" urltoken ", text)
    text = PHONE.sub(" phonetoken ", text)
    text = EMOJI.sub(" ", text)
    text = re.sub(r"[^\w\s.,!?;:()\-\"'…]", " ", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    text = re.sub(r"\.{4,}", "...", text)
    return re.sub(r"\s+", " ", text).strip()


def mask_names(text):
    """Che username và một cụm tên 2–4 từ ở đầu câu; không hứa ẩn danh hoàn toàn."""
    text = USERNAME.sub("user_name", str(text))
    tokens = text.split()
    count = 0
    while count < min(4, len(tokens)):
        token = tokens[count]
        if not (token.isalpha() and token[0].isupper() and token[1:].islower()):
            break
        count += 1
    for length in range(count, 1, -1):
        words = [strip_accents(w) for w in tokens[:length]]
        # Không dùng riêng dấu hiệu viết hoa: cần họ và tất cả thành tố thuộc danh sách.
        if (words[0] in SURNAMES or words[-1] in SURNAMES) and all(w in NAME_WORDS for w in words):
            return ("user_name " + " ".join(tokens[length:])).strip()
    return text


def preprocess(cleaned, original=None, remove_stop=False):
    original = cleaned if original is None else original
    has_emoji = bool(EMOJI.search(str(cleaned)) or EMOJI.search(str(original)))
    text = mask_names(clean_text(cleaned)).lower()
    text = REPEAT_LETTER.sub(r"\1\1", text)
    tokens = word_tokenize(text, format="text").split() if text else []
    # underthesea có thể ghép các token đặc biệt với từ bên cạnh; giữ đúng đầu ra đó.
    if remove_stop:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    if has_emoji:
        tokens.append("emoji_token")
    return " ".join(tokens)
