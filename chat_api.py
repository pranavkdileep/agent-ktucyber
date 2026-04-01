import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import AsyncIterator, Iterable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route


NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "",
)

BASE_DIR = Path(__file__).resolve().parent
STRUCTURES_DIR = BASE_DIR / "structures"
IMAGEDATA_DIR = BASE_DIR / "imagedata"
EXTRACTED_PAGES_DIR = BASE_DIR / "extracted_pages"
EXTRACTED_FIGURES_DIR = BASE_DIR / "extracted_figures"


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def remove_fields(data, fields: list[str] | None = None):
    fields = fields or []
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    if isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def get_image_objects_by_page_numbers(
    page_numbers: int | Iterable[int],
    json_path: str | Path,
) -> list[dict]:
    if isinstance(page_numbers, int):
        requested_pages = {page_numbers}
    else:
        requested_pages = {int(page_number) for page_number in page_numbers}

    with Path(json_path).open("r", encoding="utf-8") as file:
        image_objects = json.load(file)

    return [
        image_object
        for image_object in image_objects
        if image_object.get("page_number") in requested_pages
    ]


def normalize_node_id(raw_node_id: object, *, width: int = 4) -> str:
    value = str(raw_node_id).strip()
    if value.isdigit():
        return value.zfill(width)
    return value


def extract_first_json_object(text: str) -> dict:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


def iter_structure_nodes(structure: object):
    if isinstance(structure, dict):
        if structure.get("node_id") is not None:
            yield structure
        for child in structure.get("nodes", []) or []:
            yield from iter_structure_nodes(child)
        return
    if isinstance(structure, list):
        for item in structure:
            yield from iter_structure_nodes(item)


def build_node_map_from_structure(tree: dict) -> dict[str, dict]:
    structure = tree.get("structure", tree)
    mapping: dict[str, dict] = {}
    for node in iter_structure_nodes(structure):
        node_id = node.get("node_id")
        if not node_id:
            continue
        mapping[str(node_id)] = {
            "node": node,
            "start_index": node.get("start_index"),
            "end_index": node.get("end_index"),
        }
    return mapping


def guess_media_type(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(str(path))
    return media_type or "application/octet-stream"


def create_openai_client(api_key: str, base_url: str):
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is not installed in the current Python environment."
        ) from exc
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)


async def call_nvidia_vlm_once(
    prompt: str,
    image_paths: list[str] | None = None,
    model: str = "qwen/qwen3.5-397b-a17b",
    api_key: str = NVIDIA_API_KEY,
    base_url: str = NVIDIA_BASE_URL,
) -> str:
    if not api_key:
        raise RuntimeError("Missing NVIDIA_API_KEY.")

    client = create_openai_client(api_key=api_key, base_url=base_url)
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]

    for image_path in image_paths or []:
        image_file_path = Path(image_path)
        if not image_file_path.exists():
            continue
        with image_file_path.open("rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            }
        )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        stream=False,
    )
    return response.choices[0].message.content.strip()


async def stream_nvidia_vlm(
    prompt: str,
    image_paths: list[str] | None = None,
    model: str = "openai/gpt-oss-120b",
    api_key: str = NVIDIA_API_KEY,
    base_url: str = NVIDIA_BASE_URL,
) -> AsyncIterator[str]:
    if not api_key:
        raise RuntimeError("Missing NVIDIA_API_KEY.")

    client = create_openai_client(api_key=api_key, base_url=base_url)
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]

    for image_path in image_paths or []:
        image_file_path = Path(image_path)
        if not image_file_path.exists():
            continue
        with image_file_path.open("rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            }
        )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        stream=True,
    )

    async for chunk in response:
        delta = chunk.choices[0].delta
        chunk_content = getattr(delta, "content", "") or ""
        if chunk_content:
            yield chunk_content


async def search_query(query: str, tree: dict) -> dict:
    tree_without_text = remove_fields(tree.copy(), fields=["text"])
    search_prompt = f"""
    You are given a question and a tree structure of a document.
    Each node contains a node id, node title, and a corresponding summary.
    Your task is to find all tree nodes that are likely to contain the answer to the question.
    Only consider the content in the node titles and summaries, do not consider the full text of the document.
    only return relevent nodes that are likely to contain the answer to the question 1 to 3 number of nodes are ok, do not return irrelevant nodes.

    Question: {query}

    Document tree structure:
    {json.dumps(tree_without_text, indent=2)}

    Please reply in the following JSON format:
    {{
        "thinking": "<Your thinking process on which nodes are relevant to the question>",
        "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """

    tree_search_result = await call_nvidia_vlm_once(
        prompt=search_prompt,
        model="openai/gpt-oss-120b",
    )
    tree_search_result_json = extract_first_json_object(tree_search_result)

    node_map = build_node_map_from_structure(tree)
    page_hits: list[dict] = []
    final_page_range: list[str] = []
    missing_node_ids: list[str] = []

    for raw_node_id in tree_search_result_json.get("node_list", []):
        node_id = normalize_node_id(raw_node_id)
        node_info = node_map.get(node_id)
        if node_info is None and node_id.isdigit():
            node_info = node_map.get(str(int(node_id)).zfill(4))
        if node_info is None:
            missing_node_ids.append(str(raw_node_id))
            continue

        node = node_info["node"]
        start_page = node_info["start_index"]
        end_page = node_info["end_index"]
        page_range = start_page if start_page == end_page else f"{start_page}-{end_page}"
        final_page_range.append(str(page_range))
        page_hits.append(
            {
                "node_id": str(node["node_id"]),
                "title": node.get("title"),
                "start_page": int(start_page),
                "end_page": int(end_page),
                "page_range": str(page_range),
            }
        )

    if not final_page_range:
        raise RuntimeError("No matching nodes were found for the query; cannot derive a page range.")

    selected_range = final_page_range[0]
    start_page = int(selected_range.split("-")[0])
    end_page = int(selected_range.split("-")[-1]) if "-" in selected_range else start_page

    return {
        "thinking": tree_search_result_json.get("thinking", ""),
        "node_list": tree_search_result_json.get("node_list", []),
        "page_hits": page_hits,
        "missing_node_ids": missing_node_ids,
        "start_page": start_page,
        "end_page": end_page,
    }


def load_book_tree(bookid: str) -> dict:
    structure_path = STRUCTURES_DIR / f"{bookid}.json"
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")
    with structure_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_markdown_content(bookid: str, start_page: int, end_page: int) -> str:
    pages_dir = EXTRACTED_PAGES_DIR / bookid
    if not pages_dir.exists():
        raise FileNotFoundError(f"Extracted pages folder not found: {pages_dir}")

    parts: list[str] = []
    for page in range(start_page, end_page + 1):
        page_path = pages_dir / f"page_{page}.md"
        if page_path.exists():
            parts.append(page_path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def load_image_objects(bookid: str, start_page: int, end_page: int) -> list[dict]:
    imagedata_path = IMAGEDATA_DIR / f"{bookid}.json"
    if not imagedata_path.exists():
        raise FileNotFoundError(f"Image metadata file not found: {imagedata_path}")
    return get_image_objects_by_page_numbers(
        range(start_page, end_page + 1),
        json_path=imagedata_path,
    )


def build_answer_prompt(query: str, bookid: str, markdown_content: str, imageobjects: list[dict]) -> str:
    figures_dir = EXTRACTED_FIGURES_DIR / bookid
    image_list = [
        (
            f"ID: {img['figure_id']}, Caption: {img.get('image_caption_text', 'N/A')}, "
            f"File: {figures_dir / (str(img['figure_id']) + '.png')}"
        )
        for img in imageobjects
    ]

    return f"""
        You are an expert educator. Generate a well-structured, study notebook for the following query: "{query}".

        Use the provided textbook content (in markdown) as the primary source of information.

        Textbook Content:
        {markdown_content}

        Available Images:
        {image_list}

        Instructions:
        1. Organize the note with clear headings, subheadings, bullet points, and tables where appropriate.
        2. Explain complex clearly and concisely, using examples if necessary.
        3. Include relevant images from the "Available Images" list by inserting the tag <image id="ImageID"> at the most appropriate place in the text.
        4. Do NOT make up image IDs. Only use IDs from the provided list.
        5. If an image is relevant, place it near the text that explains it.
        6. Use markdown
        7. The output should be pure markdown (except for the <image> tags).
        8. The output should like a actual study note not like ai generated answer, it should be concise and to the point, avoid unnecessary explanations or verbose language.
        """


async def health(_: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def list_bookids(_: Request) -> Response:
    bookids = sorted(path.stem for path in STRUCTURES_DIR.glob("*.json"))
    return JSONResponse({"bookids": bookids})


def resolve_image_path(bookid: str, image_id: str) -> Path:
    return EXTRACTED_FIGURES_DIR / bookid / f"{image_id}.png"


def find_image_path_by_id(image_id: str) -> Path | None:
    matches = list(EXTRACTED_FIGURES_DIR.glob(f"*/{image_id}.png"))
    if not matches:
        return None
    return matches[0]


async def chat_stream(request: Request) -> Response:
    if request.method != "POST":
        return JSONResponse({"error": "Method not allowed."}, status_code=405)

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Request body must be valid JSON."}, status_code=400)

    query = str(payload.get("query", "")).strip()
    bookid = str(payload.get("bookid", "")).strip()
    if not query or not bookid:
        return JSONResponse({"error": "Both 'query' and 'bookid' are required."}, status_code=400)

    async def event_stream() -> AsyncIterator[str]:
        final_answer = ""
        try:
            yield sse_event("status", {"stage": "loading_book", "bookid": bookid})
            tree = load_book_tree(bookid)

            yield sse_event("status", {"stage": "searching_pages", "query": query})
            search_result = await search_query(query, tree)
            yield sse_event("search_result", search_result)

            start_page = search_result["start_page"]
            end_page = search_result["end_page"]
            yield sse_event(
                "status",
                {"stage": "loading_content", "start_page": start_page, "end_page": end_page},
            )

            markdown_content = load_markdown_content(bookid, start_page, end_page)
            imageobjects = load_image_objects(bookid, start_page, end_page)
            image_payload = [
                {
                    "figure_id": image["figure_id"],
                    "page_number": image["page_number"],
                    "image_caption_text": image.get("image_caption_text"),
                    "image_url": f"/books/{bookid}/images/{image['figure_id']}",
                }
                for image in imageobjects
            ]
            yield sse_event("images", {"items": image_payload})

            answer_prompt = build_answer_prompt(query, bookid, markdown_content, imageobjects)
            yield sse_event("status", {"stage": "generating_answer"})

            async for chunk in stream_nvidia_vlm(answer_prompt):
                final_answer += chunk
                yield sse_event("answer_chunk", {"delta": chunk})

            yield sse_event(
                "complete",
                {
                    "bookid": bookid,
                    "query": query,
                    "start_page": start_page,
                    "end_page": end_page,
                    "final_answer": final_answer.strip(),
                },
            )
        except Exception as exc:
            yield sse_event("error", {"message": str(exc)})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


async def get_image(request: Request) -> Response:
    bookid = request.path_params["bookid"]
    image_id = str(request.path_params["image_id"])
    image_path = resolve_image_path(bookid, image_id)
    if not image_path.exists():
        return JSONResponse({"error": "Image not found."}, status_code=404)
    return FileResponse(image_path, media_type=guess_media_type(image_path))


async def get_image_by_id(request: Request) -> Response:
    image_id = str(request.path_params["image_id"])
    image_path = find_image_path_by_id(image_id)
    if image_path is None or not image_path.exists():
        return JSONResponse({"error": "Image not found."}, status_code=404)
    return FileResponse(image_path, media_type=guess_media_type(image_path))


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/bookids", list_bookids, methods=["GET"]),
    Route("/chat/stream", chat_stream, methods=["POST"]),
    Route("/images/{image_id:str}", get_image_by_id, methods=["GET"]),
    Route("/books/{bookid:str}/images/{image_id:str}", get_image, methods=["GET"]),
]

app = Starlette(routes=routes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chat_api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
