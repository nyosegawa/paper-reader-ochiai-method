import os
import base64
import cv2
import numpy as np
import arxiv
from openai import OpenAI
import gradio as gr
from io import BytesIO
import pdf2image
import tempfile
import layoutparser as lp
from datetime import datetime
from pdfminer.high_level import extract_text
from dotenv import load_dotenv

load_dotenv()

# 環境変数からAPIキーを取得
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# OpenAIクライアントの初期化
openai_client = OpenAI(api_key=openai_api_key)


def download_paper(arxiv_url: str, save_dir: str) -> str:
    """
    arXivから論文をダウンロードする関数

    Args:
        arxiv_url (str): ダウンロードする論文のarXivのURL
        save_dir (str): 論文を保存するディレクトリのパス

    Returns:
        str: ダウンロードした論文のPDFファイルのパス
    """
    paper_id = arxiv_url.split("/")[-1]
    result = arxiv.Search(id_list=[paper_id])
    paper = next(result.results())

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{paper_id}.pdf"
    pdf_path = os.path.join(save_dir, filename)
    paper.download_pdf(dirpath=save_dir, filename=filename)

    return pdf_path


def extract_figures_and_tables(pdf_path: str, save_dir: str) -> list:
    """
    PDFから図表を抽出する関数

    Args:
        pdf_path (str): 図表を抽出するPDFファイルのパス
        save_dir (str): 抽出した画像を保存するディレクトリのパス

    Returns:
        list: 抽出した図表の情報を格納したリスト
    """
    model = lp.Detectron2LayoutModel(
        "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    )

    images = pdf2image.convert_from_path(pdf_path)

    figure_and_table_data = []
    os.makedirs(save_dir, exist_ok=True)

    for i, image in enumerate(images):
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        layout = model.detect(image_np)

        for j, block in enumerate(layout):
            if block.type in ["Table", "Figure"]:
                segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(
                    image_np
                )
                image_path = os.path.join(save_dir, f"page_{i}_block_{j}.jpg")
                cv2.imwrite(
                    image_path, segment_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )
                with open(image_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")
                figure_and_table_data.append(
                    {"path": image_path, "base64": base64_image, "type": block.type}
                )

    return figure_and_table_data


def extract_formulas(pdf_path: str, save_dir: str) -> list:
    """
    PDFから数式を抽出する関数

    Args:
        pdf_path (str): 数式を抽出するPDFファイルのパス
        save_dir (str): 抽出した画像を保存するディレクトリのパス

    Returns:
        list: 抽出した数式の情報を格納したリスト
    """
    model = lp.Detectron2LayoutModel(
        "lp://MFD/faster_rcnn_R_50_FPN_3x/config",
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map={1: "Equation"},
    )

    images = pdf2image.convert_from_path(pdf_path)

    figure_and_table_data = []
    os.makedirs(save_dir, exist_ok=True)

    for i, image in enumerate(images):
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        layout = model.detect(image_np)

        for j, block in enumerate(layout):
            if block.type in ["Equation"]:
                segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(
                    image_np
                )
                image_path = os.path.join(save_dir, f"page_{i}_block_{j}.jpg")
                cv2.imwrite(
                    image_path, segment_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )
                with open(image_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")
                figure_and_table_data.append(
                    {"path": image_path, "base64": base64_image, "type": block.type}
                )

    return figure_and_table_data


def pdf_to_base64(pdf_path: str) -> list:
    """
    PDFをbase64エンコードされた画像のリストに変換する関数

    Args:
        pdf_path (str): 変換するPDFファイルのパス

    Returns:
        list: base64エンコードされた画像のリスト
    """
    images = pdf2image.convert_from_path(pdf_path)

    base64_images = []

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="jpeg")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)

    return base64_images


def generate_image_explanation(image: str, pdf_text: str) -> str:
    """
    画像の説明を生成する関数

    Args:
        image (str): base64エンコードされた画像
        pdf_text (str): 論文から抽出したテキスト

    Returns:
        str: 生成された画像の説明
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"あなたは優秀な研究者です。論文から抽出したテキスト情報は以下です:\n{pdf_text}\n\n提供された論文の画像の示す意味を説明してください。",
            },
            {
                "role": "user",
                "content": [
                    "これは論文から抽出した画像です",
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "説明はMarkdown形式かつ日本語で記述してください。",
                    },
                ],
            },
        ],
    )
    print(response.usage)
    return response.choices[0].message.content


def generate_formula_explanation(image: str, pdf_text: str) -> str:
    """
    数式の説明を生成する関数

    Args:
        image (str): base64エンコードされた数式の画像
        pdf_text (str): 論文から抽出したテキスト

    Returns:
        str: 生成された数式の説明
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"あなたは優秀な研究者です。論文から抽出したテキスト情報は以下です:\n{pdf_text}\n\n提供された論文の数式部分の画像を提供するので、この数式の解説を行ってください",
            },
            {
                "role": "user",
                "content": [
                    "これは論文から抽出した画像です",
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{image}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "text",
                        "text": "数式はmarkdown内で使えるmathjaxを用い$$で囲んでください。解説はMarkdown形式かつ日本語で記述してください。Markdownは```で囲まないでください",
                    },
                ],
            },
        ],
    )
    print(response.usage)
    return response.choices[0].message.content


def generate_paper_summary_ochiai(images: list, arxiv_url: str) -> str:
    """
    落合メソッドで論文の要約を生成する関数

    Args:
        images (list): base64エンコードされた論文の画像のリスト
        arxiv_url (str): 論文のarXivのURL

    Returns:
        str: 生成された論文の要約
    """
    start = datetime.now()
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """あなたは優秀な研究者です。提供された論文の画像を元に以下のフォーマットに従って論文の解説を行ってください。

# {論文タイトル}

date: {YYYY-MM-DD}
categories: {論文のカテゴリ}

## 1. どんなもの？
## 2. 先行研究と比べてどこがすごいの？
## 3. 技術や手法の"キモ"はどこにある？
## 4. どうやって有効だと検証した？
## 5. 議論はあるか？
## 6. 次に読むべき論文はあるか？
## 7. 想定される質問と回答
## 論文情報・リンク
- [著者，"タイトル，" ジャーナル名 voluem no.，ページ，年](論文リンク)
""",
            },
            {
                "role": "user",
                "content": [
                    f"論文URL: {arxiv_url}",
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        images,
                    ),
                    {
                        "type": "text",
                        "text": "論文の解説はMarkdown形式かつ日本語で記述してください。",
                    },
                ],
            },
        ],
    )
    end = datetime.now()
    print('Time:', end-start)
    print('ochiai_gpt4o_withimage:')
    print(response.usage)
    return response.choices[0].message.content

def generate_paper_summary_ochiai_text(pdf_text: str, arxiv_url: str) -> str:
    """
    落合メソッドで論文の要約を生成する関数

    Args:
        pdf_text (str): 論文から抽出したテキスト
        arxiv_url (str): 論文のarXivのURL

    Returns:
        str: 生成された論文の要約
    """
    start = datetime.now()
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """あなたは優秀な研究者です。提供された論文の画像を元に以下のフォーマットに従って論文の解説を行ってください。

# {論文タイトル}

date: {YYYY-MM-DD}
categories: {論文のカテゴリ}

## 1. どんなもの？
## 2. 先行研究と比べてどこがすごいの？
## 3. 技術や手法の"キモ"はどこにある？
## 4. どうやって有効だと検証した？
## 5. 議論はあるか？
## 6. 次に読むべき論文はあるか？
## 7. 想定される質問と回答
## 論文情報・リンク
- [著者，"タイトル，" ジャーナル名 voluem no.，ページ，年](論文リンク)
""",
            },
            {
                "role": "user",
                "content": f"""論文URL: {arxiv_url}
以下が論文の内容です

---
{pdf_text}
---
論文の解説はMarkdown形式かつ日本語で記述してください。""",
            },
        ],
    )
    end = datetime.now()
    print('Time:', end-start)
    print('ochiai_gpt4o_withtext:')
    print(response.usage)
    return response.choices[0].message.content

def paper_reader(arxiv_url: str, processing_mode: str) -> tuple:
    """
    論文を読み、要約と説明を生成する関数

    Args:
        arxiv_url (str): 論文のarXivのURL
        processing_mode (str): 処理モード（"all", "formula_only", "none"のいずれか）

    Returns:
        tuple: 生成された論文の要約、数式・図表の説明、画像の説明のリスト
    """
    formatted_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir_name = os.path.join(tempfile.mkdtemp(), formatted_date)

    pdf_path = download_paper(arxiv_url, save_dir_name)
    pdf_text = extract_text(pdf_path)
    images = pdf_to_base64(pdf_path)
    paper_summary_ochiai = generate_paper_summary_ochiai_text(pdf_text, arxiv_url)
    # paper_summary_ochiai = generate_paper_summary_ochiai(images, arxiv_url)

    gallery_data = []
    if processing_mode == "none":
        explaination_text = ""
    else:
        formula_data = extract_formulas(pdf_path, save_dir_name)

        explaination_text = "# 数式の説明\n\n"
        for i, data in enumerate(formula_data):
            explanation = generate_formula_explanation(data["base64"], pdf_text)
            gallery_data.append([data["path"], explanation])
            explaination_text += f"## 数式画像{i}\n\n![](data:image/jpg;base64,{data['base64']})\n\n{explanation}\n\n"
        if processing_mode != "formula_only":
            figures_and_tables_data = extract_figures_and_tables(
                pdf_path, save_dir_name
            )
            explaination_text += "# 図表の説明\n\n"
            for i, data in enumerate(figures_and_tables_data):
                explanation = generate_image_explanation(data["base64"], pdf_text)
                gallery_data.append([data["path"], explanation])
                explaination_text += f"## 画像{i}\n\n![](data:image/jpg;base64,{data['base64']})\n\n{explanation}\n\n"

    return paper_summary_ochiai, explaination_text, gallery_data


# Gradioインターフェースの設定
demo = gr.Interface(
    fn=paper_reader,
    inputs=[
        gr.Textbox(
            label="論文URL (arXiv)", placeholder="例: https://arxiv.org/abs/2405.16153"
        ),
        gr.Radio(
            choices=[
                ("数式, 図表の解説を行う", "all"),
                ("数式の解説のみ行う", "formula_only"),
                ("数式, 図表の解説を行わない", "none"),
            ],
            label="処理方式",
            value="all",
        ),
    ],
    outputs=[
        gr.Markdown(label="落合メソッドでの解説", show_label=True),
        gr.Markdown(label="数式, 図表の解説", show_label=True),
        gr.Gallery(label="画像説明", show_label=True, elem_id="gallery"),
    ],
    title="論文の解説を落合メソッドで生成するアプリ",
)

if __name__ == "__main__":
    demo.launch(share=True)
