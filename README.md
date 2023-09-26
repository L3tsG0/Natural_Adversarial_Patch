# Natural_Adversarial_Patch

## 説明

poetryをインストールして([参考url](https://qiita.com/ksato9700/items/b893cf1db83605898d8a))、
```poetry install```を実行するとライブラリがインストールされます。

最初の方はコミットのたびにライブラリが増える可能性があるので、pullするたびに
```poetry install```をして、ライブラリをインストールしないといけないかも。

[Kaggle](https://www.kaggle.com/datasets/veeralakrishna/butterfly-dataset)からデータセットをダウンロードして、```data/leedsbutterfly```ディレクトリの下に```image```と```segmentations```を配置してください。

```poetry run python src/make_moth_data.py```を実行すると```data/leedsbutterfly/cropped```下に蝶のデータセットが作られます。
