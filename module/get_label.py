from pathlib import Path
import csv

def get_int2label_dict(path: Path):
    """モデルの整数のラベルから文字列のラベルに変換する辞書を得る"""
    int2label = {}
    with open(path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)

        for row in csv_reader:
            key = int(row[0])
            value = row[1]
            int2label[key] = value

        return int2label
