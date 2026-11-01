"""
ai_assisted_spam_ham.py
------------------
Sử dụng mô hình GLM thông qua OpenRouter API để tự động dự đoán nhãn spam/ham
cho toàn bộ tập dữ liệu bình luận (comments_cleaned.csv), sau đó tải lên Argilla
dưới dạng `rg.Suggestion` (AI-Assisted Labeling).

Tính năng:
  - Gọi OpenRouter API với mô hình GLM (mặc định: z-ai/glm-4.7-flash hoặc z-ai/glm-5.2:free).
  - Gom cụm (batching) nhiều đánh giá vào 1 prompt để giảm chi phí và tăng tốc độ gấp 10-15 lần.
  - Hỗ trợ đa luồng (concurrent workers) xử lý song song.
  - Tự động lưu checkpoint (cache) vào file JSON, cho phép dừng và tiếp tục (resume) mà không mất dữ liệu.
  - Đẩy trực tiếp các gợi ý (Suggestion) kèm điểm tin cậy (Confidence Score) lên Argilla.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import argilla as rg

# Tắt cảnh báo không cần thiết
warnings.filterwarnings("ignore", category=UserWarning, module="argilla")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

# Fix UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH MẶC ĐỊNH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = BASE_DIR / "data" / "comments_cleaned.csv"
DEFAULT_CACHE_PATH = BASE_DIR / "data" / "predictions_glm_cache.json"

DEFAULT_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-4.7-flash"  # Tốc độ cao, rẻ, hoặc dùng "z-ai/glm-5.2:free"
DEFAULT_ARGILLA_URL = "http://localhost:6900"
DEFAULT_ARGILLA_KEY = "admin.apikey"
DEFAULT_WORKSPACE = "argilla"
DEFAULT_DATASET = "comments_spamham"

VALID_LABELS = {"spam", "ham"}

SYSTEM_PROMPT = """Bạn là chuyên gia gán nhãn dữ liệu cho bài toán Phát hiện Spam (Spam Detection) tiếng Việt cho các bình luận trong các nhóm Facebook của Học viện Công nghệ Bưu chính Viễn thông (PTIT).
Nhiệm vụ của bạn là phân loại từng bình luận thành ĐÚNG 1 trong 2 nhãn sau:
- "ham": Bình luận hợp lệ, thật, liên quan đến thảo luận trong nhóm (hỏi đáp về học tập, khóa học, chia sẻ kinh nghiệm, hỏi thăm bạn bè), do người dùng thật viết.
- "spam": Nội dung rác/quảng cáo (chào mời khóa học/sản phẩm/dịch vụ như Aptis, GPA, chỉ tiêu xét tuyển...), chèn link hoặc số điện thoại đáng nghi, mời chào cờ bạc/vay tiền/việc làm, bình luận lặp lại hàng loạt (copy-paste), bot/tài khoản ảo, nội dung vô nghĩa hoặc không liên quan gì đến thảo luận.

LƯU Ý VỀ TOKEN TIỀN XỬ LÝ:
- "urltoken": bình luận gốc có chứa URL (đường dẫn) — đây là tín hiệu spam mạnh, hãy cân nhắc kỹ.
- "phonetoken": bình luận gốc có chứa số điện thoại — đây cũng là tín hiệu spam mạnh, hãy cân nhắc kỹ.
Các token này thay thế nội dung cụ thể của URL/SĐT trong văn bản đã làm sạch.

QUY TẮC BẮT BUỘC:
Chỉ trả về MỘT mảng JSON hợp lệ (JSON Array). KHÔNG kèm bất kỳ lời dẫn giải hay văn bản nào ngoài JSON.
Mỗi phần tử có đúng cấu trúc:
[
  {"id": "<id_bản_ghi>", "label": "spam|ham", "confidence": <float từ 0.5 đến 1.0>}
]
""".strip()


def load_env_file():
    """Tự động đọc OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_ID từ file .env nếu có."""
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


def parse_args():
    load_env_file()

    env_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    env_base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or DEFAULT_OPENAI_BASE_URL
    )
    env_model = os.environ.get("MODEL_ID") or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
    env_dataset = os.environ.get("DATASET") or DEFAULT_DATASET
    env_argilla_url = os.environ.get("ARGILLA_API_URL") or DEFAULT_ARGILLA_URL
    env_argilla_key = os.environ.get("ARGILLA_API_KEY") or DEFAULT_ARGILLA_KEY

    parser = argparse.ArgumentParser(description="AI-Assisted Labeling với mô hình GLM qua API tương thích OpenAI.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=env_api_key,
        help="API Key (mặc định lấy từ OPENAI_API_KEY trong .env)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=env_base_url,
        help=f"Base URL tương thích OpenAI (mặc định lấy từ OPENAI_BASE_URL trong .env: {env_base_url})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=env_model,
        help=f"ID mô hình (mặc định lấy từ MODEL_ID trong .env: {env_model})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=env_dataset,
        help=f"Tên dataset, ID hoặc URL trên Argilla (mặc định lấy từ DATASET trong .env: {env_dataset})",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"File CSV dữ liệu đầu vào (mặc định: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(DEFAULT_CACHE_PATH),
        help=f"File cache lưu kết quả dự đoán (mặc định: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Số lượng đánh giá gom trong 1 lần gọi API (mặc định: 15)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Số luồng gọi API song song (mặc định: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số bản ghi cần dự đoán (để trống nếu muốn chạy toàn bộ)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        default=True,
        help="Tải kết quả lên Argilla sau khi dự đoán (mặc định: True)",
    )
    parser.add_argument(
        "--no-upload",
        dest="upload",
        action="store_false",
        help="Chỉ dự đoán và lưu vào cache, không đẩy lên Argilla",
    )
    parser.add_argument(
        "--reset-dataset",
        action="store_true",
        help="Xóa và tạo mới dataset trên Argilla trước khi tải lên",
    )
    parser.add_argument(
        "--auto-submit-threshold",
        type=float,
        default=None,
        help="Tự động chốt thành nhãn chính thức (submitted response) nếu confidence >= ngưỡng này (ví dụ: 0.75)",
    )
    parser.add_argument(
        "--argilla-url",
        type=str,
        default=DEFAULT_ARGILLA_URL,
        help=f"URL server Argilla (mặc định: {DEFAULT_ARGILLA_URL})",
    )
    parser.add_argument(
        "--argilla-key",
        type=str,
        default=DEFAULT_ARGILLA_KEY,
        help=f"Argilla API Key (mặc định: {DEFAULT_ARGILLA_KEY})",
    )
    return parser.parse_args()


def load_cache(cache_path: Path) -> dict:
    """Tải các kết quả đã dự đoán trước đó để có thể tiếp tục (resume)."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache_path: Path, cache_data: dict):
    """Lưu cache ra file an toàn."""
    temp_path = cache_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    temp_path.replace(cache_path)


def clean_json_response(raw_text) -> str:
    """Loại bỏ thẻ ```json ... ``` hoặc thẻ <think>...</think> nếu LLM bao bọc."""
    if not raw_text:
        return ""
    text = str(raw_text).strip()
    # 1. Loại bỏ thẻ suy nghĩ <think>...</think> nếu có từ các reasoning model
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Tìm khối code markdown ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()

    # 3. Trích xuất chính xác mảng JSON [ ... ] nếu có văn bản ngoài lề
    match_arr = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
    if match_arr:
        return match_arr.group(0).strip()

    return text


def predict_batch_with_glm(batch_items, api_key: str, model: str, base_url: str, max_retries: int = 3):
    """Gửi 1 batch đánh giá tới endpoint tương thích OpenAI bằng mô hình GLM."""
    prompt_lines = ["Dưới đây là danh sách các bình luận cần phân loại spam/ham:\n"]
    for item in batch_items:
        prompt_lines.append(f'- ID: {item["id"]} | Nội dung: "{item["text"]}"')

    prompt_content = "\n".join(prompt_lines)

    endpoint_url = base_url.rstrip("/")
    if not endpoint_url.endswith("/chat/completions"):
        endpoint_url = f"{endpoint_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter" in base_url.lower():
        headers["HTTP-Referer"] = "http://localhost:6900"
        headers["X-Title"] = "Argilla Spam/Ham AI Annotation"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content},
        ],
        "temperature": 0.1,
        "max_tokens": 3500,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                endpoint_url,
                headers=headers,
                json=payload,
                timeout=90,
            )
            if resp.status_code == 200:
                res_json = resp.json()
                choices = res_json.get("choices", [])
                if not choices:
                    tqdm.write(f"[WARN] API trả về 200 nhưng danh sách choices trống: {res_json}")
                    time.sleep(2)
                    continue

                msg = choices[0].get("message", {})
                content = msg.get("content")
                # Nếu content là None (thường gặp ở reasoning model), trích xuất từ reasoning_content
                if not content:
                    content = msg.get("reasoning_content") or msg.get("reasoning") or ""

                cleaned_content = clean_json_response(content)
                if not cleaned_content:
                    finish_reason = choices[0].get("finish_reason")
                    tqdm.write(
                        f"[WARN] Mô hình trả về nội dung rỗng (finish_reason: {finish_reason}). "
                        "Có thể do chạm giới hạn max_tokens. Đang thử lại..."
                    )
                    time.sleep(2)
                    continue

                parsed = json.loads(cleaned_content)
                if isinstance(parsed, list):
                    # Chuẩn hóa nhãn
                    valid_results = {}
                    for p in parsed:
                        if isinstance(p, dict) and "id" in p and "label" in p:
                            lbl = str(p.get("label", "")).strip().lower()
                            if lbl in VALID_LABELS:
                                # Mặc định 0.5 nếu LLM không trả về confidence để buộc duyệt thủ công
                                conf = float(p.get("confidence", 0.5))
                                valid_results[str(p["id"])] = {
                                    "label": lbl,
                                    "confidence": min(max(conf, 0.5), 1.0),
                                }
                    if valid_results:
                        return valid_results
                    else:
                        tqdm.write(f"[WARN] Không tìm thấy nhãn hợp lệ trong JSON: {parsed[:2]}")
            elif resp.status_code == 429:
                try:
                    err_msg = resp.json().get("error", {}).get("message", "")
                except Exception:
                    err_msg = resp.text[:100]
                tqdm.write(f"[WARN] Rate Limit (429): {err_msg}. Đang đợi {4 * (attempt + 1)}s...")
                time.sleep(4 * (attempt + 1))
            else:
                tqdm.write(f"[WARN] API lỗi HTTP {resp.status_code}: {resp.text[:100]}")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            tqdm.write(f"[WARN] Lỗi kết nối mạng: {e}. Đang thử lại...")
            time.sleep(2 * (attempt + 1))
        except (json.JSONDecodeError, ValueError) as e:
            tqdm.write(f"[WARN] Lỗi đọc JSON phản hồi: {e}. Đang thử lại...")
            time.sleep(2)
        except Exception as e:
            tqdm.write(f"[WARN] Lỗi xử lý: {e}. Đang thử lại...")
            time.sleep(2 * (attempt + 1))

    return {}


def parse_dataset_identifier(dataset_str: str):
    """
    Trích xuất UUID hoặc tên dataset từ chuỗi đầu vào:
    - Nếu là URL (vd: http://localhost:6900/dataset/77ead8f4-3357-417d-bbe9-0acbe7af4c6b):
      trích xuất chuỗi UUID 36 ký tự.
    - Nếu là chuỗi UUID 36 ký tự: trả về (uuid, None)
    - Nếu là tên (vd: reviews_dienmayxanh): trả về (None, name)
    """
    if not dataset_str:
        return None, None
    dataset_str = dataset_str.strip()
    uuid_match = re.search(
        r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        dataset_str,
    )
    if uuid_match:
        return uuid_match.group(1), None
    return None, dataset_str


def setup_argilla_dataset(client, dataset_identifier: str, reset: bool = False):
    """Đảm bảo dataset trên Argilla tồn tại với cấu hình phù hợp (tìm theo ID, URL hoặc Name)."""
    target_id, target_name = parse_dataset_identifier(dataset_identifier)

    dataset = None
    if target_id:
        print(f"[*] Đang tìm dataset theo ID '{target_id}' trên Argilla...")
        dataset = next((d for d in client.datasets if str(d.id).lower() == target_id.lower()), None)
        if dataset:
            print(f"[+] Đã tìm thấy dataset: '{dataset.name}' (ID: {dataset.id})")
        else:
            print(f"[WARN] Không tìm thấy dataset với ID '{target_id}' trên server.")
    elif target_name:
        print(f"[*] Đang tìm dataset theo tên '{target_name}' trên Argilla...")
        dataset = next((d for d in client.datasets if d.name.lower() == target_name.lower()), None)
        if dataset is None:
            dataset = client.datasets(name=target_name, workspace=DEFAULT_WORKSPACE)

    if reset and dataset is not None:
        print(f"[*] Đang xóa dataset cũ '{dataset.name}' (ID: {dataset.id}) theo cờ --reset-dataset...")
        dataset.delete()
        dataset = None

    if dataset is None:
        name_to_use = target_name if target_name else DEFAULT_DATASET
        print(f"[*] Đang tạo mới dataset '{name_to_use}' trên Argilla...")
        settings = rg.Settings(
            guidelines="""
## Hướng dẫn thẩm định nhãn Spam/Ham:
AI đã gợi ý sẵn nhãn cho bạn. Hãy kiểm tra:
- **ham**: Bình luận hợp lệ, thật, liên quan đến thảo luận trong nhóm (hỏi đáp, chia sẻ kinh nghiệm).
- **spam**: Nội dung rác/quảng cáo, chào mời khóa học/dịch vụ, chứa link/SĐT đáng nghi, bình luận lặp lại hàng loạt, nội dung vô nghĩa hoặc không liên quan.
Bấm phím tắt hoặc click để xác nhận hoặc sửa nhãn.
""".strip(),
            fields=[
                rg.TextField(
                    name="cleaned_text",
                    title="Nội dung bình luận (đã làm sạch)",
                    required=True,
                    use_markdown=False,
                ),
            ],
            questions=[
                rg.LabelQuestion(
                    name="label",
                    title="Phân loại (Spam/Ham)",
                    labels=["spam", "ham"],
                    required=True,
                ),
            ],
        )

        dataset = rg.Dataset(
            name=name_to_use,
            workspace=DEFAULT_WORKSPACE,
            settings=settings,
            client=client,
        )
        dataset.create()
        print(f"[+] Đã tạo dataset '{dataset.name}' (ID: {dataset.id}) thành công!")

    return dataset


def main():
    load_env_file()
    args = parse_args()

    if not args.api_key:
        print("[ERROR] Bạn chưa cấu hình OPENAI_API_KEY!")
        print("Cách cấu hình:")
        print("  1. Thiết lập trong file .env:")
        print("       OPENAI_BASE_URL=https://openrouter.ai/api/v1")
        print("       OPENAI_API_KEY=sk-or-v1-xxx")
        print("       MODEL_ID=z-ai/glm-4.7-flash")
        print("       DATASET=http://localhost:6900/dataset/<id_dataset_spam_ham_cua_ban>")
        print("  2. Hoặc truyền tham số: python ai_assisted_glm.py --api-key sk-xxx")
        sys.exit(1)

    # 1. Kết nối Argilla và lấy Dataset
    print(f"[*] Đang kết nối tới Argilla tại {args.argilla_url}...")
    try:
        client = rg.Argilla(api_url=args.argilla_url, api_key=args.argilla_key)
        dataset = setup_argilla_dataset(client, args.dataset, reset=args.reset_dataset)
    except Exception as e:
        print(f"[ERROR] Không thể kết nối tới Argilla: {e}")
        sys.exit(1)

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_cache = load_cache(cache_path)
    print(f"[*] Đã tải cache: {len(predictions_cache):,} bản ghi đã được dự đoán trước đó.")

    # 2. Chuẩn bị dữ liệu cần xử lý (ưu tiên đọc trực tiếp từ Dataset trên Argilla)
    all_data_map = {}
    items_to_process = []

    existing_records = list(dataset.records)
    total_existing = len(existing_records)

    if total_existing > 0:
        print(f"[+] Đang sử dụng {total_existing:,} bản ghi trực tiếp từ dataset trên Argilla.")
        records_to_use = existing_records
        if args.limit and args.limit > 0:
            records_to_use = records_to_use[: args.limit]
            print(f"[*] Giới hạn số lượng xử lý: {len(records_to_use):,}")

        for rec in records_to_use:
            rid = str(rec.id)
            fields = rec.fields or {}
            text = str(fields.get("cleaned_text", "")).strip()
            all_data_map[rid] = {"id": rid, "text": text}

            has_suggestion = any(
                s.question_name == "label" for s in getattr(rec, "suggestions", [])
            )
            if rid not in predictions_cache and not has_suggestion:
                items_to_process.append({"id": rid, "text": text})
            elif has_suggestion and rid not in predictions_cache:
                for s in rec.suggestions:
                    if s.question_name == "label":
                        predictions_cache[rid] = {
                            "label": s.value,
                            "confidence": s.score or 1.0,
                        }
    else:
        # Nếu dataset trên Argilla còn trống, nạp từ CSV
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[ERROR] Dataset trên Argilla đang trống và không tìm thấy file CSV: {csv_path}")
            sys.exit(1)
        print(f"[*] Dataset trên Argilla đang trống. Đang nạp dữ liệu từ CSV: {csv_path}...")
        df = pd.read_csv(csv_path)
        if args.limit and args.limit > 0:
            df = df.head(args.limit)
        print(f"[+] Số bản ghi đọc từ CSV: {len(df):,}")

        for idx, row in df.iterrows():
            # ID sinh theo chỉ số dòng, khớp với ID bản ghi đã upload (cmt_0, cmt_1, ...)
            rid = f"cmt_{idx}"

            text_clean = str(row["cleaned_text"]).strip() if pd.notna(row.get("cleaned_text")) else ""
            text_raw = str(row["text"]).strip() if pd.notna(row.get("text")) else ""
            text = text_clean if text_clean else text_raw
            if not text:
                continue

            all_data_map[rid] = {"id": rid, "text": text}
            if rid not in predictions_cache:
                items_to_process.append({"id": rid, "text": text})

    print(f"[*] Số bản ghi mới cần gọi AI dự đoán: {len(items_to_process):,}")

    # 3. Gọi OpenRouter / OpenAI API theo batch với đa luồng
    if items_to_process:
        print(f"[*] Đang kết nối tới: {args.base_url}")
        print(f"[*] Khởi chạy mô hình '{args.model}' với {args.workers} luồng song song...")
        batches = [
            items_to_process[i : i + args.batch_size]
            for i in range(0, len(items_to_process), args.batch_size)
        ]

        completed_count = 0
        total_batches = len(batches)

        spam_count = sum(1 for v in predictions_cache.values() if v.get("label") == "spam")
        ham_count = sum(1 for v in predictions_cache.values() if v.get("label") == "ham")

        pbar = tqdm(
            total=total_batches,
            desc="[AI-Assisted] Dự đoán",
            unit="batch",
            dynamic_ncols=True,
        )
        pbar.set_postfix({
            "cache": f"{len(predictions_cache):,}",
            "spam": spam_count,
            "ham": ham_count,
        })

        interrupted = False
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        future_to_batch = {}

        try:
            future_to_batch = {
                executor.submit(
                    predict_batch_with_glm, batch, args.api_key, args.model, args.base_url
                ): batch
                for batch in batches
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_res = future.result()
                if batch_res:
                    for rid, pdata in batch_res.items():
                        predictions_cache[rid] = pdata
                        lbl = pdata.get("label")
                        if lbl == "spam":
                            spam_count += 1
                        elif lbl == "ham":
                            ham_count += 1

                completed_count += 1
                pbar.set_postfix({
                    "cache": f"{len(predictions_cache):,}",
                    "spam": spam_count,
                    "ham": ham_count,
                })
                pbar.update(1)

                # Lưu cache định kỳ mỗi 5 batch
                if completed_count % 5 == 0 or completed_count == total_batches:
                    save_cache(cache_path, predictions_cache)

        except KeyboardInterrupt:
            interrupted = True
            pbar.close()
            print("\n" + "!" * 65)
            print("[!] ĐÃ NHẬN TÍN HIỆU DỪNG (Ctrl + C) TỪ BÀN PHÍM!")
            print("[*] Đang dừng các luồng và lưu bộ nhớ đệm an toàn...")
            for f in future_to_batch:
                f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        finally:
            pbar.close()
            save_cache(cache_path, predictions_cache)
            print(f"[+] Đã lưu an toàn {len(predictions_cache):,} bản ghi vào cache: {cache_path}")
            if interrupted:
                print("!" * 65)

        if interrupted:
            if len(predictions_cache) > 0 and args.upload:
                try:
                    ans = input("\n👉 Bạn có muốn tải phần đã dự đoán lên Argilla ngay bây giờ không? [Y/n]: ").strip().lower()
                    if ans in ("n", "no"):
                        print("[*] Đã bỏ qua tải lên. Dữ liệu đã lưu trong cache, bạn có thể chạy lại để tiếp tục.")
                        sys.exit(0)
                except (KeyboardInterrupt, EOFError):
                    print("\n[*] Đã thoát chương trình an toàn.")
                    sys.exit(0)
            else:
                sys.exit(0)
        else:
            print(f"[+] Hoàn tất dự đoán! Đã lưu tổng cộng {len(predictions_cache):,} bản ghi vào cache.")

    # 4. Tải lên Argilla với Suggestion
    if args.upload:
        print(f"\n[*] Đang đóng gói bản ghi kèm gợi ý AI (rg.Suggestion)...")
        argilla_records = []
        for rid, item in all_data_map.items():
            suggestions = []
            responses = []
            if rid in predictions_cache:
                pred = predictions_cache[rid]
                suggestions.append(
                    rg.Suggestion(
                        question_name="label",
                        value=pred["label"],
                        score=pred["confidence"],
                        agent=f"openrouter/{args.model}",
                    )
                )
                if (
                    args.auto_submit_threshold is not None
                    and pred["confidence"] >= args.auto_submit_threshold
                ):
                    responses.append(
                        rg.Response(
                            question_name="label",
                            value=pred["label"],
                            user_id=client.me.id,
                            status="submitted",
                        )
                    )

            record = rg.Record(
                id=rid,
                fields={
                    "cleaned_text": item["text"],
                },
                suggestions=suggestions,
                responses=responses,
                _dataset=dataset,
            )
            argilla_records.append(record)

        try:
            print(f"[*] Bắt đầu cập nhật {len(argilla_records):,} bản ghi lên dataset '{dataset.name}'...")
            dataset.records.log(records=argilla_records, batch_size=256)
            print("\n" + "=" * 65)
            print(f"[SUCCESS] Đã cập nhật thành công {len(argilla_records):,} bản ghi lên Argilla!")
            print(f"👉 Mở ngay giao diện gán nhãn: {args.argilla_url}/dataset/{dataset.id}/annotation-mode")
            print("=" * 65)
        except KeyboardInterrupt:
            print("\n[!] Đã dừng quá trình tải lên Argilla (Ctrl + C). Các batch đã gửi trước đó vẫn được lưu.")
            sys.exit(0)


if __name__ == "__main__":
    main()


