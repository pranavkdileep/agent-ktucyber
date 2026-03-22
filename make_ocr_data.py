import argparse
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any
import time

SAMPLE_DEEPSEEK_OCR_OUTPUT = """<|ref|>image<|/ref|><|det|>[[120, 63, 875, 330]]<|/det|>
<|ref|>image_caption<|/ref|><|det|>[[118, 342, 597, 359]]<|/det|>
<center>Fig. 1.3 Basic building blocks of a generic biometric system. </center>

<|ref|>text<|/ref|><|det|>[[120, 405, 876, 443]]<|/det|>
face) while designing the sensor module. Furthermore, factors like cost, size, and durability also impact the sensor design.

<|ref|>title<|/ref|><|det|>[[120, 489, 483, 509]]<|/det|>
#### 1.2.3 Feature extraction module

<|ref|>image<|/ref|><|det|>[[120, 63, 875, 330]]<|/det|>
<|ref|>image_caption<|/ref|><|det|>[[118, 342, 597, 359]]<|/det|>
<center>Fig. 1.3333333 Basic building blocks of a generic biometric system. </center>

<|ref|>text<|/ref|><|det|>[[118, 533, 878, 880]]<|/det|>
Usually, the raw biometric data from the sensor is subjected to pre- processing operations before features are extracted from it. The three commonly used pre- processing steps are (a) quality assessment, (b) segmentation, and (c) enhancement. First, the quality of the acquired biometric samples needs to be accessed to determine its suitability for further processing.
"""

OCR_HEADER_PATTERN = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|><\|det\|>(?P<bbox>\[\[.*?\]\])<\|/det\|>",
    re.DOTALL,
)


def load_pymupdf():
    try:
        import pymupdf
    except ModuleNotFoundError as exc:
        try:
            import fitz  # type: ignore
        except Exception as fitz_exc:
            raise RuntimeError(
                "PyMuPDF is required to render PDF pages, but this environment has "
                "the unrelated `fitz` package instead, which triggers the misleading "
                "`Directory 'static/' does not exist` error. "
                "Fix it with `pip uninstall fitz && pip install PyMuPDF`."
            ) from fitz_exc

        fitz_file = Path(getattr(fitz, "__file__", ""))
        if "site-packages/fitz/" in fitz_file.as_posix():
            raise RuntimeError(
                "Imported `fitz` from the unrelated `fitz` package, which causes "
                "the misleading `Directory 'static/' does not exist` error. "
                "Uninstall `fitz`, then install PyMuPDF with `pip uninstall fitz && pip install PyMuPDF`."
            ) from exc

        return fitz

    return pymupdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PDF pages, run dummy DeepSeek OCR, and append figure metadata to JSON.",
    )
    parser.add_argument("pdf_file", help="Path to the input PDF file.")
    parser.add_argument("start_page", type=int, help="1-based start page number.")
    parser.add_argument("end_page", type=int, help="1-based end page number.")
    parser.add_argument(
        "--output",
        default="ocr_figures.json",
        help="Path to the JSON output file. Default: ocr_figures.json",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render DPI for page images. Default: 200",
    )
    return parser.parse_args()


def get_pdf_page_count(pdf_path: Path) -> int:
    pymupdf = load_pymupdf()
    with pymupdf.open(pdf_path) as pdf_document:
        return len(pdf_document)


def validate_page_range(start_page: int, end_page: int, total_pages: int) -> None:
    if start_page < 1 or end_page < 1:
        raise ValueError("start_page and end_page must be >= 1.")
    if start_page > end_page:
        raise ValueError("start_page must be less than or equal to end_page.")
    if end_page > total_pages:
        raise ValueError(f"end_page cannot be greater than total pages ({total_pages}).")


def render_page_to_image(pdf_path: Path, page_number: int, temp_dir: Path, dpi: int) -> Path:
    pymupdf = load_pymupdf()
    pdf_document = pymupdf.open(pdf_path)
    try:
        zero_based_page = page_number - 1
        if zero_based_page < 0 or zero_based_page >= len(pdf_document):
            raise ValueError(f"Invalid page number {page_number} for PDF with {len(pdf_document)} pages.")

        page = pdf_document.load_page(zero_based_page)
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(temp_dir, f"page_{page_number}.jpg")

        with open(image_path, "wb") as image_file:
            image_file.write(pix.tobytes("jpeg"))

        print(f"Saved page {page_number} image: {image_path}")
        return Path(image_path)
    finally:
        pdf_document.close()


def run_dummy_deepseek_ocr(image_path: str) -> str:
    # Replace this with the real OCR/model call later.
    text_result = model.infer(
        tokenizer,
        prompt="<image>\n<|grounding|>Convert the document to markdown. ",
        image_file=image_path,
        output_path='/kaggle/working/ocr',
        base_size=config["base_size"],
        image_size=config["image_size"],
        crop_mode=config["crop_mode"],
        save_results=True,
        test_compress=True,
        eval_mode=True,
    )
    #print(text_result)
    return text_result


def clean_caption_text(text: str) -> str:
    text = re.sub(r"</?center>", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()


def parse_bbox(raw_bbox: str) -> list[int]:
    numbers = [int(value) for value in re.findall(r"-?\d+", raw_bbox)]
    if len(numbers) != 4:
        raise ValueError(f"Expected 4 bbox values, got {numbers}")
    return numbers


def extract_figure_entries(ocr_output: str, page_number: int) -> list[dict[str, Any]]:
    blocks = []
    matches = list(OCR_HEADER_PATTERN.finditer(ocr_output))
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(ocr_output)
        blocks.append(
            {
                "label": match.group("label").strip(),
                "bbox": parse_bbox(match.group("bbox")),
                "content": ocr_output[match.end():next_start].strip(),
            }
        )

    figures: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block["label"] != "image":
            continue

        caption_block = None
        if index + 1 < len(blocks) and blocks[index + 1]["label"] == "image_caption":
            caption_block = blocks[index + 1]

        figures.append(
            {
                "figure_id": random.randint(100000, 999999),
                "page_number": page_number,
                "bounding_box": block["bbox"],
                "image_caption_text": clean_caption_text(caption_block["content"]) if caption_block else "",
                "caption_bounding_box": caption_block["bbox"] if caption_block else [],
            }
        )

    return figures


def load_existing_records(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Existing JSON in {output_path} must contain a list.")
    return data


def append_records(output_path: Path, new_records: list[dict[str, Any]]) -> None:
    all_records = load_existing_records(output_path)
    all_records.extend(new_records)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_records, file, indent=2, ensure_ascii=False)


def process_pdf(pdf_path: Path, start_page: int, end_page: int, output_path: Path, dpi: int) -> list[dict[str, Any]]:
    total_pages = get_pdf_page_count(pdf_path)
    validate_page_range(start_page, end_page, total_pages)

    collected_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ocr_pages_", dir="/tmp") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for page_number in range(start_page, end_page + 1):
            image_path = render_page_to_image(pdf_path, page_number, temp_dir, dpi)
            ocr_output = run_dummy_deepseek_ocr(str(image_path))
            page_records = extract_figure_entries(ocr_output, page_number)
            if page_records:
                append_records(output_path, page_records)
                collected_records.extend(page_records)

    return collected_records


def main() -> None:
    #args = parse_args()
    pdf_path = Path("/kaggle/input/datasets/pranavkdileepme/demoimage/textbook.pdf")
    output_path = Path("output.json")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    records = process_pdf(
        pdf_path=pdf_path,
        start_page=1,
        end_page=100,
        output_path=output_path,
        dpi=100,
    )
    print(f"Saved {len(records)} figure records to {output_path}")


if __name__ == "__main__":
    main()
