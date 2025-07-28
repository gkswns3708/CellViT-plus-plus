# make_splits.py
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv

# 1) Train 이미지 스템(stem) 목록 수집
img_dir = Path("dataset/transformed/CoNSeP/Train/images")
stems = [p.stem for p in img_dir.glob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]]

# 2) 80% train / 20% val 분할
train, val = train_test_split(
    stems,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

# 3) splits/fold_0 에 CSV 저장
out_dir = Path("dataset/transformed/CoNSeP/splits/fold_0")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir/"train.csv","w",newline="") as f:
    writer=csv.writer(f)
    for name in train:
        writer.writerow([name])

with open(out_dir/"val.csv","w",newline="") as f:
    writer=csv.writer(f)
    for name in val:
        writer.writerow([name])

print(f"✔️ train: {len(train)}, val: {len(val)} saved to {out_dir}")
