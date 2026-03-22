import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def load_pymupdf():
    try:
        import pymupdf
    except ModuleNotFoundError as exc:
        try:
            import fitz  # type: ignore
        except Exception as fitz_exc:
            raise RuntimeError(
                "PyMuPDF is required to extract figure crops, but this environment has "
                "the unrelated `fitz` package instead, which triggers the misleading "
                "`Directory 'static/' does not exist` error. "
                "Fix it with `pip uninstall fitz && pip install PyMuPDF`."
            ) from fitz_exc

        fitz_file = Path(getattr(fitz, "__file__", ""))
        if "site-packages/fitz/" in fitz_file.as_posix():
            raise RuntimeError(
                "Imported `fitz` from the unrelated `fitz` package. "
                "Fix it with `pip uninstall fitz && pip install PyMuPDF`."
            ) from exc

        return fitz

    return pymupdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PDF pages and save cropped figure images from JSON bounding boxes.",
    )
    parser.add_argument("pdf_file", help="Path to the source PDF.")
    parser.add_argument("json_file", help="Path to the JSON file containing figure records.")
    parser.add_argument("output_dir", help="Directory where cropped figure images will be saved.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render DPI used for the saved crops. Default: 200",
    )
    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=("jpg", "png"),
        help="Output image format. Default: jpg",
    )
    return parser.parse_args()


def load_records(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {json_path}")
    return data


def validate_record(record: dict[str, Any], index: int) -> tuple[str, int, tuple[int, int, int, int]]:
    figure_id = str(record.get("figure_id", "")).strip()
    page_number = record.get("page_number")
    bbox = record.get("bounding_box")

    if not figure_id:
        raise ValueError(f"Record {index} is missing a valid `figure_id`.")
    if not isinstance(page_number, int) or page_number < 1:
        raise ValueError(f"Record {index} has invalid `page_number`: {page_number!r}")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Record {index} has invalid `bounding_box`: {bbox!r}")

    try:
        bbox_values = tuple(int(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Record {index} has non-numeric `bounding_box`: {bbox!r}") from exc

    return figure_id, page_number, bbox_values


def parse_optional_bbox(record: dict[str, Any], field_name: str) -> tuple[int, int, int, int] | None:
    bbox = record.get(field_name)
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        return tuple(int(value) for value in bbox)
    except (TypeError, ValueError):
        return None


def clamp_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size
    left = min(max(left, 0), width)
    top = min(max(top, 0), height)
    right = min(max(right, 0), width)
    bottom = min(max(bottom, 0), height)
    return left, top, right, bottom


def scale_bbox_from_thousand_space(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1_norm, y1_norm, x2_norm, y2_norm = bbox
    width, height = image_size
    x1 = int(x1_norm / 1000 * width)
    y1 = int(y1_norm / 1000 * height)
    x2 = int(x2_norm / 1000 * width)
    y2 = int(y2_norm / 1000 * height)
    return x1, y1, x2, y2


def extract_figures(
    pdf_path: Path,
    records: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
    image_format: str,
) -> int:
    pymupdf = load_pymupdf()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    with pymupdf.open(pdf_path) as pdf_document:
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        rendered_pages: dict[int, Image.Image] = {}

        for index, record in enumerate(records):
            figure_id, page_number, bbox = validate_record(record, index)
            if page_number not in rendered_pages:
                page = pdf_document.load_page(page_number - 1)
                pix = page.get_pixmap(matrix=matrix)
                rendered_pages[page_number] = Image.frombytes(
                    "RGB",
                    (pix.width, pix.height),
                    pix.samples,
                )

            page_image = rendered_pages[page_number]
            crop_bbox = scale_bbox_from_thousand_space(bbox, page_image.size)
            crop_bbox = clamp_bbox(crop_bbox, page_image.size)
            left, top, right, bottom = crop_bbox
            if right <= left or bottom <= top:
                raise ValueError(
                    f"Record {index} produced an empty crop for figure_id={figure_id} on page {page_number}."
                )

            output_path = output_dir / f"{figure_id}.{image_format}"
            figure_crop = page_image.crop(crop_bbox)

            caption_bbox = parse_optional_bbox(record, "caption_bounding_box")
            if caption_bbox is not None:
                caption_crop_bbox = scale_bbox_from_thousand_space(caption_bbox, page_image.size)
                caption_crop_bbox = clamp_bbox(caption_crop_bbox, page_image.size)
                cap_left, cap_top, cap_right, cap_bottom = caption_crop_bbox

                if cap_right > cap_left and cap_bottom > cap_top:
                    caption_crop = page_image.crop(caption_crop_bbox)
                    combined_width = max(figure_crop.width, caption_crop.width)
                    combined_height = figure_crop.height + caption_crop.height
                    combined_image = Image.new("RGB", (combined_width, combined_height), "white")
                    combined_image.paste(figure_crop, (0, 0))
                    combined_image.paste(caption_crop, (0, figure_crop.height))
                    figure_crop = combined_image

            figure_crop.save(output_path)
            saved_count += 1
            print(f"Saved {output_path}")

    return saved_count


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_file).expanduser().resolve()
    json_path = Path(args.json_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    records = load_records(json_path)
    saved_count = extract_figures(
        pdf_path=pdf_path,
        records=records,
        output_dir=output_dir,
        dpi=100,
        image_format=args.image_format,
    )
    print(f"Saved {saved_count} figure crops to {output_dir}")


if __name__ == "__main__":
    main()
