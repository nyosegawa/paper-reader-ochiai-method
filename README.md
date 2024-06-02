![](https://github.com/nyosegawa/paper-reader-ochiai-method/assets/116951203/e9b4c7d1-bc97-4b89-a841-163eddad1f80)

解説記事: [Qiita](https://qiita.com/sakasegawa/items/8e17ede26dd96e7e3280)

## 環境準備

```
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

`.env` を作成
https://platform.openai.com/api-keys でAPIキーを取得し貼り付け

```
OPENAI_API_KEY=sk...
```

## 使い方

```
python app.py
```

## 注意

- 数式、図表の解説を行う場合、1本あたり100-300円くらいかかります
    - 数式・図表がめちゃ入ってる場合はもっとかかります
