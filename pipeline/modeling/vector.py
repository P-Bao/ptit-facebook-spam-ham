import numpy as np
import math
import re

docs = [
    "xin xin lịch thi",
    "xin lịch học",
    "nhận làm thuê"
]

# Bước 1: Tách từ
cac_cau = []

for doc in docs:
    tokens = re.findall(r"\w+", doc.lower())
    cac_cau.append(tokens)

# Bước 2: Tạo từ điển
tu_dien = sorted(set(
    tu for cau in cac_cau for tu in cau
))

N = len(docs)
V = len(tu_dien)

# Bước 3: Tạo ma trận BoW
bow = np.zeros((N, V))

for i in range(N):
    for j in range(V):
        bow[i][j] = cac_cau[i].count(tu_dien[j])

# Bước 4: Tính TF-IDF
tfidf = np.zeros((N, V))

for j in range(V):
    # DF: số câu chứa từ này
    df = 0

    for cau in cac_cau:
        if tu_dien[j] in cau:
            df += 1

    idf = 1 + math.log((N + 1) / (df + 1))

    for i in range(N):
        tfidf[i][j] = bow[i][j] * idf

print("Từ điển:", tu_dien)
print("BoW:\n", bow)
print("TF-IDF:\n", np.round(tfidf, 3))