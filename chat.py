import base64
import json
import os
from pathlib import Path
from typing import Iterable
import pageindex.utils as utils
import fitz
import openai


OUTPUT_JSON_PATH = Path(__file__).with_name("output.json")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

STRUCTURES_DIR = Path("structures")
IMAGEDATA_DIR = Path("imagedata")
EXTRACTED_PAGES_DIR = Path("extracted_pages")
EXTRACTED_FIGURES_DIR = Path("extracted_figures")


def get_image_objects_by_page_numbers(
    page_numbers: int | Iterable[int],
    json_path: str | Path = OUTPUT_JSON_PATH,
) -> list[dict]:
    """
    Return image objects from output.json for the given page number(s).

    Args:
        page_numbers: A single page number or an iterable of page numbers.
        json_path: Path to the JSON file containing image objects.

    Returns:
        A list of matching image objects.
    """
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


async def call_nvidia_vlm(
    prompt: str,
    image_paths: list[str] | None = None,
    model: str = "qwen/qwen3.5-397b-a17b", #qwen/qwen3.5-397b-a17b
    api_key: str = NVIDIA_API_KEY,
    base_url: str = NVIDIA_BASE_URL,
    stream: bool = False,
) -> str:
    """
    Call a vision-language model using NVIDIA's OpenAI-compatible endpoint.

    Args:
        prompt: The text prompt to send to the model.
        image_paths: Optional list of local image paths to include.
        model: The vision model name exposed by the endpoint.
        api_key: API key for the NVIDIA endpoint.
        base_url: Base URL for the OpenAI-compatible endpoint.

    Returns:
        The model's text response.
    """
    if not api_key:
        raise RuntimeError(
            "Missing NVIDIA_API_KEY. Set it in the environment, e.g. `export NVIDIA_API_KEY=...`"
        )
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]

    for image_path in image_paths or []:
        if not os.path.exists(image_path):
            continue

        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            }
        )

    # response = await client.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content": content}],
    #     temperature=0,
    # )
    # return response.choices[0].message.content.strip()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        stream=stream,
    )
    if stream:
        full_response = ""
        async for chunk in response:
            delta = chunk.choices[0].delta
            chunk_content = getattr(delta, "content", "") or ""
            full_response += chunk_content
            print(chunk_content, end="", flush=True)
        print()  # for newline after streaming is done
        return full_response.strip()
    else:
        return response.choices[0].message.content.strip()


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

async def search_query(query:str,tree:dict):
    tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])
    #print(json.dumps(tree_without_text, indent=2))
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

    tree_search_result = await call_nvidia_vlm(prompt=search_prompt, model="openai/gpt-oss-120b")
    print("Tree search result:", tree_search_result)
    tree_search_result_json = extract_first_json_object(tree_search_result)

    node_map = build_node_map_from_structure(tree)
    final_page_range = []
    missing_node_ids: list[str] = []
    for raw_node_id in tree_search_result_json.get("node_list", []):
        node_id = normalize_node_id(raw_node_id)

        node_info = node_map.get(node_id)
        if node_info is None and node_id.isdigit():
            node_info = node_map.get(str(int(node_id)).zfill(4))
        if node_info is None:
            missing_node_ids.append(str(raw_node_id))
            continue

        node = node_info['node']
        start_page = node_info['start_index']
        end_page = node_info['end_index']
        page_range = start_page if start_page == end_page else f"{start_page}-{end_page}"
        print(f"Node ID: {node['node_id']}\t Pages: {page_range}\t Title: {node['title']}")
        final_page_range.append(page_range)
    if not final_page_range:
        raise RuntimeError("No matching nodes were found for the query; cannot derive a page range.")
    print(final_page_range[0])
    start_page = final_page_range[0].split("-")[0]
    end_page = final_page_range[0].split("-")[-1] if "-" in final_page_range[0] else start_page
    if missing_node_ids:
        print("Warning: node_ids not found in tree:", missing_node_ids)
    return start_page, end_page

async def main():
    query = "Feature Extraction module in biometric"
    bookid = "BIOMETRIC-SECURITY"

    structure_path = STRUCTURES_DIR / f"{bookid}.json"
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    with structure_path.open("r", encoding="utf-8") as file:
        tree = json.load(file)

    start, end = await search_query(query, tree)
    print(f"Start page: {start}, End page: {end}")

    imagedata_path = IMAGEDATA_DIR / f"{bookid}.json"
    if not imagedata_path.exists():
        raise FileNotFoundError(f"Image metadata file not found: {imagedata_path}")

    imageobjects = get_image_objects_by_page_numbers(
        range(int(start), int(end) + 1),
        json_path=imagedata_path,
    )
    print(f"Image objects on pages {start}-{end}:")
    for image_object in imageobjects:
        print(f"Page {image_object['page_number']}: {image_object['figure_id']}")
    print("Generating answer using VLM...")
    markdown_content = ""

    pages_dir = EXTRACTED_PAGES_DIR / bookid
    if not pages_dir.exists():
        raise FileNotFoundError(f"Extracted pages folder not found: {pages_dir}")

    for page in range(int(start), int(end) + 1):
        pagepath = pages_dir / f"page_{page}.md"
        if not pagepath.exists():
            continue
        markdown_content += pagepath.read_text(encoding="utf-8") + "\n"

    figures_dir = EXTRACTED_FIGURES_DIR / bookid
    imageList = [
        (
            f"ID: {img['figure_id']}, Caption: {img.get('image_caption_text', 'N/A')}, "
            f"File: {figures_dir / (str(img['figure_id']) + '.png')}"
        )
        for img in imageobjects
    ]
    answer_prompt = f"""
        You are an expert educator. Generate a well-structured, study notebook for the following query: "{query}".
        
        Use the provided textbook content (in markdown) as the primary source of information.
        
        Textbook Content:
        {markdown_content}
        
        Available Images:
        {imageList}
        
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

    # If you want to actually send the images to the VLM, you can pass:
    # image_paths=[str(figures_dir / f"{img['figure_id']}.png") for img in imageobjects]
    final_answer = await call_nvidia_vlm(answer_prompt, stream=True)
    print("Generated Study Note:\n")
    print(final_answer)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
