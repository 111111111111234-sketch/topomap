import openai
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps
import base64
from io import BytesIO
import os
import time
import re
import math
import json
from hashlib import sha256
from typing import Optional
import logging
from omegaconf import OmegaConf
from src.const import *
from src.vlm_runtime import (
    begin_vlm_call,
    finish_vlm_call,
    get_vlm_client,
    get_vlm_runtime,
    record_vlm_attempt,
)

DEFAULT_VLM_MODEL = "openai/gpt-4o"


class NonRetryableVLMError(RuntimeError):
    """A provider rejected an unchanged request payload."""


def _decode_base64_image(image_base64: str) -> Image.Image:
    """Decode a prompt image while normalizing it for contact-sheet assembly."""
    return Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")


def _encode_image_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _label_font(size: int = 17):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def make_labeled_contact_sheet(items, tile_size: int = 180, max_cols: int = 5) -> str:
    """Combine labeled base64 images into one image without changing their IDs.

    Some OpenAI-compatible providers accept only four image parts per request.
    The tile label is deliberately rendered into the pixels so the VLM can still
    return the original ``Snapshot i, Object j`` / ``Frontier i`` indices.
    """
    if not items:
        raise ValueError("Cannot create a contact sheet without images")

    padding = 6
    label_height = 30
    cols = min(max_cols, len(items))
    rows = (len(items) + cols - 1) // cols
    tile_height = tile_size + label_height
    sheet = Image.new(
        "RGB",
        (cols * (tile_size + 2 * padding), rows * (tile_height + 2 * padding)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = _label_font()

    for idx, (label, image_base64) in enumerate(items):
        row, col = divmod(idx, cols)
        x = col * (tile_size + 2 * padding) + padding
        y = row * (tile_height + 2 * padding) + padding
        image = ImageOps.contain(_decode_base64_image(image_base64), (tile_size, tile_size))
        image_x = x + (tile_size - image.width) // 2
        image_y = y + (tile_size - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        draw.rectangle(
            (x, y + tile_size, x + tile_size, y + tile_height), fill=(0, 0, 0)
        )
        # Keep the index visible even if a detector produces an unusually long class name.
        draw.text((x + 3, y + tile_size + 6), label[:42], fill=(255, 255, 255), font=font)

    return _encode_image_base64(sheet)


def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{c[1]}",
                        "detail": "high",
                    },
                }
            )
    return formated_content


# send information to openai
def call_openai_api(
    sys_prompt,
    contents,
    model_name: str = DEFAULT_VLM_MODEL,
    extra_body: Optional[dict] = None,
    purpose: str = "goat_explorer",
    raise_bad_request: bool = False,
    max_tries: int = 5,
) -> Optional[str]:
    max_tries = max(1, int(max_tries))
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    started_at = begin_vlm_call(purpose)
    while retry_count < max_tries:
        try:
            record_vlm_attempt(purpose)
            request_kwargs = dict(
                model=model_name,
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            client = get_vlm_client()
            completion = client.chat.completions.create(**request_kwargs)
            choice = completion.choices[0]
            message = choice.message
            content = message.content
            if content:
                finish_vlm_call(started_at, True, purpose)
                return content

            # Some Qwen-compatible gateways place visible text in a nonstandard
            # reasoning field. Use it as a fallback, then log enough metadata to
            # distinguish an empty model completion from a transport failure.
            model_extra = getattr(message, "model_extra", {}) or {}
            if get_vlm_runtime()["reasoning_content_fallback"]:
                for field in ("reasoning_content", "reasoning", "analysis", "text"):
                    fallback = getattr(message, field, None) or model_extra.get(field)
                    if isinstance(fallback, str) and fallback.strip():
                        logging.warning(
                            "VLM returned content through %s instead of message.content", field
                        )
                        finish_vlm_call(started_at, True, purpose)
                        return fallback
            logging.warning(
                "VLM returned an empty completion (model=%s, finish_reason=%s, extra_fields=%s)",
                model_name,
                getattr(choice, "finish_reason", None),
                sorted(model_extra.keys()),
            )
            finish_vlm_call(started_at, False, purpose)
            return None
        except openai.BadRequestError as e:
            logging.error(
                "VLM bad request (model=%s, purpose=%s, image_parts=%d): %s",
                model_name,
                purpose,
                sum(1 for item in formated_content if item.get("type") == "image_url"),
                e,
            )
            finish_vlm_call(started_at, False, purpose)
            if raise_bad_request:
                raise NonRetryableVLMError(str(e)) from e
            return None
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 3s")
            time.sleep(3)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(3)
            retry_count += 1
            continue

    finish_vlm_call(started_at, False, purpose)
    return None


def strip_thinking_content(response: str) -> str:
    """Remove inline Qwen thinking blocks if an endpoint returns them in content."""
    if "</think>" in response:
        final_answer = response.split("</think>", 1)[1].lstrip()
        # Do not turn a response containing only a thinking block into an
        # unhelpful empty string; the structured-choice fallback can still
        # recover an answer that appears inside that block.
        return final_answer if final_answer else response
    return response


def extract_structured_choice(response: str) -> Optional[str]:
    """Recover an HGR action if a model wraps it in extra reasoning text."""
    match = re.search(
        r"\b(?:snapshot\s+\d+\s*,\s*object\s+\d+|frontier\s+\d+)\b",
        response,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return re.sub(r"\s+", " ", match.group(0)).replace(" ,", ",")


# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img).convert("RGB")
    width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot encode an empty prompt image: {width}x{height}")

    # Qwen3-VL validates every Base64 image independently. Detector crops can
    # be only a few pixels wide or have extreme aspect ratios, which makes an
    # otherwise small prompt fail as a provider-side 400. Preserve pixels and
    # aspect ratio by padding (not stitching candidates) onto a valid canvas.
    min_pixels = 65536
    max_aspect_ratio = 10.0
    aspect_ratio = max(width / height, height / width)
    if width * height < min_pixels or aspect_ratio > max_aspect_ratio:
        padded_width = max(width, int(math.ceil(height / max_aspect_ratio)))
        padded_height = max(height, int(math.ceil(width / max_aspect_ratio)))
        padded = Image.new("RGB", (padded_width, padded_height), color=(0, 0, 0))
        padded.paste(img, ((padded_width - width) // 2, (padded_height - height) // 2))
        fitted = ImageOps.contain(padded, (256, 256), method=Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (256, 256), color=(0, 0, 0))
        canvas.paste(fitted, ((256 - fitted.width) // 2, (256 - fitted.height) // 2))
        img = canvas
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal


def get_step_info(
    step,
    verbose=False,
    model_name: str = DEFAULT_VLM_MODEL,
    extra_body: Optional[dict] = None,
    max_images: Optional[int] = None,
):
    # 1 get question data
    question, image_goal = format_question(step)

    # 2 get step information(egocentric, frontier, snapshot)
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))

    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))

    # 2.2.1 get frontier semantic predictions (Hypothesis Graph Refinement)
    frontier_semantics = step.get("frontier_semantic_predictions", None)

    # 2.3 get snapshots
    snapshot_classes = {}  # rgb_id -> list of classes
    snapshot_full_imgs = {}  # rgb_id -> full img
    snapshot_crops = {}  # rgb_id -> list of crops
    snapshot_clusters = {}  # rgb_id -> list of clusters
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]["full_img"]
        snapshot_full_imgs[rgb_id] = encode_tensor2base64(snapshot_img)
        snapshot_crops[rgb_id] = [
            encode_tensor2base64(crop_data["crop"])
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        snapshot_class = [
            crop_data["obj_class"]
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        cluster_class = [
            obj_map[int(obj_id)] for obj_id in step["snapshot_objects"][rgb_id]
        ]
        # remove duplicates
        seen_classes.update(sorted(list(set(snapshot_class))))
        snapshot_classes[rgb_id] = snapshot_class
        snapshot_clusters[rgb_id] = cluster_class

    # 3 prefiltering, note that we need the obj_id_mapping
    snapshot_source_indices = list(step.get(
        "snapshot_source_indices", range(len(snapshot_full_imgs))
    ))
    if len(snapshot_source_indices) != len(snapshot_full_imgs):
        raise ValueError(
            "snapshot_source_indices must align with the exposed snapshot set"
        )
    keep_index = list(snapshot_source_indices)
    keep_index_snapshot = {
        rgb_id: list(range(len(snapshot_crops[rgb_id]))) for rgb_id in snapshot_crops
    }
    if step.get("use_prefiltering") is True:
        use_full_obj_list = step["use_full_obj_list"]
        n_prev_snapshot = len(snapshot_full_imgs)
        snapshot_classes, filtered_positions, keep_index_snapshot = prefiltering(
            question,
            snapshot_classes,
            snapshot_clusters,
            seen_classes,
            step["top_k_categories"],
            image_goal,
            use_full_obj_list,
            verbose=verbose,
            model_name=model_name,
            extra_body=extra_body,
        )
        keep_index = [snapshot_source_indices[i] for i in filtered_positions]
        snapshot_full_imgs = {
            rgb_id: snapshot_full_imgs[rgb_id] for rgb_id in keep_index_snapshot.keys()
        }
        for rgb_id in snapshot_classes.keys():
            snapshot_crops[rgb_id] = [
                snapshot_crops[rgb_id][i] for i in keep_index_snapshot[rgb_id]
            ]
        if verbose:
            logging.info(
                f"Prefiltering snapshot: {n_prev_snapshot} -> {len(snapshot_full_imgs)}"
            )

    if max_images is not None:
        (
            frontier_imgs,
            frontier_semantics,
            snapshot_full_imgs,
            snapshot_classes,
            snapshot_crops,
            keep_index,
            keep_index_snapshot,
        ) = apply_visual_image_budget(
            max_images=max_images,
            has_image_goal=image_goal is not None,
            has_egocentric_view=bool(egocentric_imgs),
            frontier_imgs=frontier_imgs,
            frontier_semantics=frontier_semantics,
            snapshot_imgs=snapshot_full_imgs,
            snapshot_classes=snapshot_classes,
            snapshot_crops=snapshot_crops,
            snapshot_id_mapping=keep_index,
            snapshot_crop_mapping=keep_index_snapshot,
        )

    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        keep_index,
        keep_index_snapshot,
        frontier_semantics,
    )


def apply_visual_image_budget(
    max_images: int,
    has_image_goal: bool,
    has_egocentric_view: bool,
    frontier_imgs,
    frontier_semantics,
    snapshot_imgs,
    snapshot_classes,
    snapshot_crops,
    snapshot_id_mapping,
    snapshot_crop_mapping,
):
    """Cap separate image parts while preserving every exposed index mapping.

    Priority is goal image, current egocentric view, frontiers, then complete
    snapshot entries (full image plus at least one selectable crop). No contact
    sheet or image stitching is used. Keeping prefixes preserves the existing
    Frontier index mapping; snapshot/object mappings are rebuilt explicitly.
    """
    max_images = int(max_images)
    if max_images < 1:
        raise ValueError("vlm_max_images must be a positive integer")

    fixed_count = int(bool(has_image_goal)) + int(bool(has_egocentric_view))
    remaining = max(0, max_images - fixed_count)
    kept_frontier_count = min(len(frontier_imgs), remaining)
    new_frontier_imgs = list(frontier_imgs[:kept_frontier_count])
    if frontier_semantics is None:
        new_frontier_semantics = None
    else:
        new_frontier_semantics = list(frontier_semantics[:kept_frontier_count])
    remaining -= kept_frontier_count

    new_snapshot_imgs = {}
    new_snapshot_classes = {}
    new_snapshot_crops = {}
    new_snapshot_id_mapping = []
    new_snapshot_crop_mapping = {}
    snapshot_items = list(snapshot_imgs.items())
    for prompt_index, (rgb_id, full_image) in enumerate(snapshot_items):
        crops = list(snapshot_crops.get(rgb_id, []))
        classes = list(snapshot_classes.get(rgb_id, []))
        crop_mapping = list(snapshot_crop_mapping.get(rgb_id, []))
        crop_count = min(len(crops), len(classes), len(crop_mapping))
        if remaining < 2 or crop_count == 0:
            continue
        kept_crop_count = min(crop_count, remaining - 1)
        new_snapshot_imgs[rgb_id] = full_image
        new_snapshot_classes[rgb_id] = classes[:kept_crop_count]
        new_snapshot_crops[rgb_id] = crops[:kept_crop_count]
        new_snapshot_id_mapping.append(snapshot_id_mapping[prompt_index])
        new_snapshot_crop_mapping[rgb_id] = crop_mapping[:kept_crop_count]
        remaining -= 1 + kept_crop_count

    before_count = (
        fixed_count
        + len(frontier_imgs)
        + len(snapshot_imgs)
        + sum(len(items) for items in snapshot_crops.values())
    )
    after_count = (
        fixed_count
        + len(new_frontier_imgs)
        + len(new_snapshot_imgs)
        + sum(len(items) for items in new_snapshot_crops.values())
    )
    if after_count > max_images:
        raise AssertionError(
            f"visual prompt budget exceeded: {after_count} > {max_images}"
        )
    if after_count < before_count:
        logging.info(
            "Visual image budget: %d -> %d parts (limit=%d, frontiers=%d, snapshots=%d)",
            before_count,
            after_count,
            max_images,
            len(new_frontier_imgs),
            len(new_snapshot_imgs),
        )
    return (
        new_frontier_imgs,
        new_frontier_semantics,
        new_snapshot_imgs,
        new_snapshot_classes,
        new_snapshot_crops,
        new_snapshot_id_mapping,
        new_snapshot_crop_mapping,
    )


def _format_active_topology_metadata(topo):
    fields = [f"state={topo['state']}"]
    if "relevance" in topo:
        fields.append(f"R={topo['relevance']:.2f}")
    fields.extend([
        f"C={topo['confidence']:.2f}",
        f"A={topo['accessibility']:.2f}",
        f"U={topo['utility']:.2f}",
    ])
    return ", ".join(fields)


def format_explore_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    snapshot_crops,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    frontier_semantics=None,
    max_prompt_images: Optional[int] = None,
    goal_topology_context: Optional[dict] = None,
    frontier_only: bool = False,
    hgr_topology_context: Optional[dict] = None,
):
    sys_prompt = (
        "Task: You are an agent in an indoor scene that is able to observe the surroundings and explore the environment. "
        + (
            "You must choose a Frontier to explore; Snapshots are visual context only.\n"
            if frontier_only else
            "You are required to choose either a Snapshot or a Frontier image to explore and find the target object required in the question.\n"
        )
    )

    content = []
    # 1 here is some basic info
    text = "Definitions:\n"
    text += (
        "Snapshot: A focused observation of several objects. It contains a full image of the cluster of objects, and separate image crops of each object. "
        + "Choosing a snapshot means that the object asked in the question is within the cluster of objects that the snapshot represents, and you will choose that object as the final answer of the question. "
        + "Therefore, if you choose a snapshot, you should also choose the object in the snapshot that you think is the answer to the question.\n"
    )
    text += "Frontier: An unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction.\n"

    # 2 here is the question
    text += f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    if hgr_topology_context is not None:
        from src.hgr_topology_context import format_hgr_topology_context
        exposed_context = {**hgr_topology_context,
                           "frontiers": hgr_topology_context["frontiers"][:len(frontier_imgs)]}
        content.append((format_hgr_topology_context(exposed_context, snapshot_imgs),))

    if goal_topology_context:
        related = goal_topology_context.get("related_observations", [])
        text = (
            "Persistent goal-conditioned topology: "
            f"the global map has {goal_topology_context.get('global_nodes', 0)} nodes "
            f"and {goal_topology_context.get('global_edges', 0)} edges; "
            f"the current connected task view has "
            f"{goal_topology_context.get('active_nodes', 0)} nodes."
        )
        if related:
            text += " Related prior observations: " + ", ".join(related) + "."
        text += (
            " Frontier indices below are ordered by goal relevance and then "
            "graph-hop proximity to this evidence; all remain real, currently "
            "executable frontiers.\n"
        )
        content.append((text,))

    text = ("Select a Frontier to explore using the snapshots as context only.\n" if frontier_only else "Select the Frontier/Snapshot that would help find the answer of the question.\n")
    content.append((text,))

    # Providers such as the selected Qwen endpoint limit an individual request
    # to four image parts. Preserve every candidate by grouping snapshots and
    # frontiers into separately labeled contact sheets. The goal and current
    # egocentric view remain individual images, so the maximum is four.
    compact_visual_prompt = max_prompt_images is not None
    if compact_visual_prompt and max_prompt_images < 4:
        raise ValueError(
            "GOAT visual prompt compaction needs max_prompt_images >= 4 "
            "when goal, egocentric, snapshot, and frontier images are present."
        )

    # 3 here is the egocentric views
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))

    # 4 here is the snapshot images
    text = ("The following snapshots are visual context for evaluating the Frontiers; do not select a Snapshot as the action.\n" if frontier_only else "The followings are all the snapshots that you can choose. Following each snapshot image are the class name and image crop of each object contained in the snapshot.\n")
    text += "Please note that the class name may not be accurate due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make the decision.\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    elif compact_visual_prompt:
        snapshot_tiles = []
        for i, rgb_id in enumerate(snapshot_imgs.keys()):
            snapshot_tiles.append((f"Snapshot {i}", snapshot_imgs[rgb_id]))
            for j in range(len(snapshot_crops[rgb_id])):
                snapshot_tiles.append(
                    (
                        f"Snapshot {i}, Object {j}: {snapshot_classes[rgb_id][j]}",
                        snapshot_crops[rgb_id][j],
                    )
                )
        content.append(
            (
                "One labeled contact sheet contains every selectable Snapshot and Object. "
                "Use the index printed on each tile exactly in your answer.\n",
                make_labeled_contact_sheet(snapshot_tiles),
            )
        )
        content.append(("\n",))
    else:
        for i, rgb_id in enumerate(snapshot_imgs.keys()):
            content.append((f"Snapshot {i} ", snapshot_imgs[rgb_id]))
            for j in range(len(snapshot_crops[rgb_id])):
                content.append(
                    (
                        f"Object {j}: {snapshot_classes[rgb_id][j]}",
                        snapshot_crops[rgb_id][j],
                    )
                )
            content.append(("\n",))

    # 5 here is the frontier images with hypothesis semantic predictions
    text = "The followings are all the Frontiers that you can explore: \n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available\n",))
    elif compact_visual_prompt:
        frontier_tiles = [
            (f"Frontier {i}", frontier_img)
            for i, frontier_img in enumerate(frontier_imgs)
        ]
        content.append(
            (
                "One labeled contact sheet contains every Frontier. "
                "Use the index printed on each tile exactly in your answer.\n",
                make_labeled_contact_sheet(frontier_tiles),
            )
        )
        for i, pred in enumerate(frontier_semantics or []):
            if pred is not None:
                cats = pred["predicted_categories"]
                probs = pred["probabilities"]
                pairs = ", ".join(
                    [f"{c} ({p:.2f})" for c, p in zip(cats, probs)]
                )
                content.append(
                    (
                        f"Frontier {i} predicted room types: {pairs}; "
                        f"uncertainty: {pred['entropy']:.2f}\n",
                    )
                )
                topo = pred.get("active_topology")
                if topo is not None:
                    content.append((
                        f"Frontier {i} topology: "
                        f"{_format_active_topology_metadata(topo)}\n",
                    ))
        content.append(("\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            # Add hypothesis semantic predictions if available
            if frontier_semantics is not None and i < len(frontier_semantics) and frontier_semantics[i] is not None:
                pred = frontier_semantics[i]
                cats = pred["predicted_categories"]
                probs = pred["probabilities"]
                pairs = ", ".join([f"{c} ({p:.2f})" for c, p in zip(cats, probs)])
                content.append((f" [Predicted room types: {pairs}, Uncertainty: {pred['entropy']:.2f}]",))
                topo = pred.get("active_topology")
                if topo is not None:
                    content.append((
                        f" [Topology: {_format_active_topology_metadata(topo)}]",
                    ))
            content.append(("\n",))

    # 6 here is the format of the answer
    text = ("Please provide your answer only in the format: 'Frontier i', where i is the Frontier index. " if frontier_only else "Please provide your answer in the following format: 'Snapshot i, Object j' or 'Frontier i', where i, j are the index of the snapshot or frontier you choose. ")
    text += (
        "Return only a Frontier index; Snapshot/object indices are not valid actions in this mode.\n"
        if frontier_only else
        "For example, if you choose the fridge in the first snapshot, please return 'Snapshot 0, Object 2', where 2 is the index of the fridge in that snapshot.\n"
    )
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    content.append((text,))

    return sys_prompt, content


def format_prefiltering_prompt(question, class_list, top_k=10, image_goal=None):
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
    prompt = "Your goal is to answer questions about the scene through exploration.\n"
    prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance.\n"
    prompt += "These are the rules for the task.\n"
    prompt += "1. Read through the whole object list.\n"
    prompt += "2. Rank objects in the list based on how well they can help your exploration given the question.\n"
    prompt += f"3. Reprint the name of all objects that may help your exploration given the question. "
    prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
    content.append((prompt,))
    # ------------------format an example-------------------------
    prompt = "Here is an example of selecting helpful objects:\n"
    prompt += "Question: What can I use to watch my favorite shows and movies?\n"
    prompt += (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    prompt += "painting\nspeaker\nbox\ncabinet\nlamp\ntv\nbook rack\nsofa\noven\nbed\ncurtain\n"
    prompt += "Answer: tv\nspeaker\nsofa\nbed\n"
    content.append((prompt,))
    # ------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt + "\n",))
    prompt = (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    for i, cls in enumerate(class_list):
        prompt += f"{cls}\n"
    prompt += "Answer: "
    content.append((prompt,))
    return sys_prompt, content


def get_prefiltering_classes(
    question,
    seen_classes,
    top_k=10,
    image_goal=None,
    model_name: str = DEFAULT_VLM_MODEL,
    extra_body: Optional[dict] = None,
):
    prefiltering_sys, prefiltering_content = format_prefiltering_prompt(
        question, sorted(list(seen_classes)), top_k=top_k, image_goal=image_goal
    )

    message = ""
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image {c[1][:10]}..."
    response = call_openai_api(
        prefiltering_sys,
        prefiltering_content,
        model_name=model_name,
        extra_body=extra_body,
        purpose="goat_prefilter",
    )
    if response is None:
        return []
    response = strip_thinking_content(response)

    # parse the response and return the top_k objects
    selected_classes = response.strip().split("\n")
    selected_classes = [cls.strip() for cls in selected_classes]
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]

    return selected_classes


def prefiltering(
    question,
    snapshot_classes,
    snapshot_clusters,
    seen_classes,
    top_k=10,
    image_goal=None,
    use_full_obj_list=False,
    verbose=False,
    model_name: str = DEFAULT_VLM_MODEL,
    extra_body: Optional[dict] = None,
):
    selected_classes = get_prefiltering_classes(
        question,
        seen_classes,
        top_k,
        image_goal,
        model_name=model_name,
        extra_body=extra_body,
    )
    if verbose:
        logging.info(f"Prefiltering selected classes: {selected_classes}")

    keep_index = [
        i
        for i, k in enumerate(snapshot_clusters.keys())
        if len(set(snapshot_clusters[k]) & set(selected_classes)) > 0
    ]
    keep_snapshot_id = [list(snapshot_classes.keys())[i] for i in keep_index]
    snapshot_classes = {rgb_id: snapshot_classes[rgb_id] for rgb_id in keep_snapshot_id}

    keep_index_snapshot = {}
    for rgb_id in keep_snapshot_id:
        keep_index_snapshot[rgb_id] = [
            i
            for i in range(len(snapshot_classes[rgb_id]))
            if snapshot_classes[rgb_id][i] in selected_classes
        ]
        snapshot_classes[rgb_id] = [
            snapshot_classes[rgb_id][i] for i in keep_index_snapshot[rgb_id]
        ]

    return snapshot_classes, keep_index, keep_index_snapshot


def explore_step(step, cfg, verbose=False):
    model_name = cfg.get("vlm_model", DEFAULT_VLM_MODEL)
    extra_body_cfg = cfg.get("vlm_extra_body", None)
    extra_body = (
        OmegaConf.to_container(extra_body_cfg, resolve=True)
        if extra_body_cfg is not None
        else None
    )
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        frontier_semantics,
    ) = get_step_info(
        step,
        verbose,
        model_name=model_name,
        extra_body=extra_body,
        max_images=cfg.get("vlm_max_images", None),
    )
    context = step.get("hgr_topology_context")
    if context and context.get("object_approach"):
        context = {**context, "snapshots": {
            image: {"objects": {f"Object {i}": {
                k: v for k, v in context["snapshots"][image]["objects"][source].items()
                if k not in ("certification", "certified_path", "route_plan", "transitions", "route_query_seconds", "route_event")}
                for i, source in enumerate(snapshot_crop_mapping[image])}}
            for image in snapshot_full_imgs}}
        context["frontiers"] = [{k:v for k,v in row.items() if k not in ("certification", "certified_path", "route_plan", "transitions", "route_query_seconds", "route_event")}
                                for row in context["frontiers"]]
    sys_prompt, content = format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        frontier_semantics=frontier_semantics,
        max_prompt_images=cfg.get("max_prompt_images", None),
        goal_topology_context=step.get("goal_topology_context"),
        frontier_only=bool(step.get("frontier_only", False)),
        hgr_topology_context=context,
    )
    if step.get("navigation_task_context") is not None:
        task_rows = {}
        for snapshot_index, image in enumerate(snapshot_full_imgs):
            for object_index, source in enumerate(snapshot_crop_mapping[image]):
                entity = str(step["snapshot_imgs"][image]["object_crop"][source]["obj_id"])
                purpose = step["navigation_task_context"].get(str(image), {}).get(entity)
                if purpose is not None:
                    task_rows[f"Snapshot {snapshot_index}, Object {object_index}"] = purpose
        content.append(("Executable object task purposes: " + json.dumps(task_rows) +
            ". APPROACH means approach the selected entity; REVISIT means obtain new evidence, "
            "not proof of goal identity. Choose based on visual goal evidence. Frontier semantics "
            "remain supplied by HGR hypotheses; arrival is not identity confirmation.",))

    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 3
    final_response = None
    final_reason = None
    for _ in range(retry_bound):
        try:
            response = call_openai_api(
                sys_prompt,
                content,
                model_name=model_name,
                extra_body=extra_body,
                purpose=step.get("request_purpose", "goat_explorer"),
                raise_bad_request=True,
            )
        except NonRetryableVLMError:
            logging.error("Explorer request rejected; skipping identical outer retries")
            break

        if response is None:
            print("call_openai_api returns None, retrying")
            continue

        raw_response = strip_thinking_content(response).strip()
        structured_choice = (
            extract_structured_choice(raw_response)
            if cfg.get("vlm_structured_choice_fallback", False)
            else None
        )
        if structured_choice is not None:
            response, reason = structured_choice, raw_response
        elif "\n" in raw_response:
            response = raw_response.split("\n")
            response, reason = response[0], response[-1]
        else:
            response = raw_response
            reason = ""
        response = response.lower()
        try:
            choice_type, choice_id = response.split(",")[0].strip().split(" ")
        except Exception as e:
            print(f"Error in splitting response: {response}")
            print(e)
            continue

        response_valid = False
        if (
            choice_type == "snapshot"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(snapshot_full_imgs)
        ):
            try:
                object_choice_type, object_choice_id = (
                    response.split(",")[1].strip().split(" ")
                )
            except Exception as e:
                print(f"Error in splitting response: {response}")
                print(e)
                continue
            if (
                object_choice_type == "object"
                and object_choice_id.isdigit()
                and 0
                <= int(object_choice_id)
                < len(list(snapshot_crop_mapping.values())[int(choice_id)])
            ):
                response_valid = True
        elif (
            choice_type == "frontier"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(frontier_imgs)
        ):
            response_valid = True

        if response_valid:
            final_response = response
            final_reason = reason
            break

    return (
        final_response,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        final_reason,
        len(snapshot_full_imgs),
    )


def assess_navigation_step(step, cfg, navigator, verbose=False):
    """Return route-blind semantic assessments for the hierarchical navigator.

    This intentionally shares HGR's filtering and visual budget implementation
    with ``explore_step``.  Topology metadata is not passed to the formatter,
    so route length and Place identity cannot influence semantic confidence.
    """
    from src.hierarchical_navigator import (
        SemanticAssessment,
        SemanticConfidence,
        parse_semantic_assessments,
    )

    model_name = cfg.get("vlm_model", DEFAULT_VLM_MODEL)
    extra_body_cfg = cfg.get("vlm_extra_body", None)
    extra_body = (
        OmegaConf.to_container(extra_body_cfg, resolve=True)
        if extra_body_cfg is not None else None
    )
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        frontier_semantics,
    ) = get_step_info(
        step,
        verbose,
        model_name=model_name,
        extra_body=extra_body,
        max_images=cfg.get("vlm_max_images", None),
    )

    candidate_ids = []
    candidate_kinds = {}
    for snapshot_index, rgb_id in enumerate(snapshot_full_imgs):
        for object_index in range(len(snapshot_crops[rgb_id])):
            candidate_id = f"Snapshot {snapshot_index}, Object {object_index}"
            candidate_ids.append(candidate_id)
            candidate_kinds[candidate_id] = "snapshot"
    for frontier_index in range(len(frontier_imgs)):
        candidate_id = f"Frontier {frontier_index}"
        candidate_ids.append(candidate_id)
        candidate_kinds[candidate_id] = "frontier"

    # Candidate labels are prompt-local, therefore include their stable source
    # mappings in the digest. Pixels are immutable for a Snapshot image ID;
    # live Frontier identity is provided by the caller.
    digest_payload = {
        "question": question,
        "goal_type": step.get("task_type"),
        "goal_image": step.get("image") if step.get("task_type") == "image" else None,
        "snapshot_ids": list(snapshot_full_imgs.keys()),
        "snapshot_mapping": snapshot_id_mapping,
        "crop_mapping": snapshot_crop_mapping,
        "snapshot_classes": snapshot_classes,
        "frontier_ids": list(step.get("frontier_stable_ids", []))[:len(frontier_imgs)],
        "frontier_semantics": frontier_semantics,
    }
    digest = sha256(
        json.dumps(digest_payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    cached = navigator.cached_assessments(digest)
    if cached is not None:
        confidence_counts = {}
        for assessment in cached.values():
            label = assessment.confidence.value
            confidence_counts[label] = confidence_counts.get(label, 0) + 1
        navigator.last_semantic_protocol = {
            "candidate_count": len(candidate_ids),
            "parse_attempts": 0,
            "fallback_used": any(
                value.evidence == "safe_default_after_missing_or_invalid_assessment"
                for value in cached.values()
            ),
            "cache_hit": True,
            "confidence_counts": confidence_counts,
            "high_candidates": [
                key for key, value in cached.items()
                if value.confidence == SemanticConfidence.HIGH
            ],
            "medium_candidates": [
                key for key, value in cached.items()
                if value.confidence == SemanticConfidence.MEDIUM
            ],
        }
        return cached, snapshot_id_mapping, snapshot_crop_mapping, len(snapshot_full_imgs), digest

    sys_prompt, content = format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        frontier_semantics=frontier_semantics,
        max_prompt_images=cfg.get("max_prompt_images", None),
        # Semantic identity must not see route cost or topology state.
        goal_topology_context=None,
        frontier_only=False,
        hgr_topology_context=None,
    )
    sys_prompt = (
        "You are the semantic candidate layer of an indoor navigation system. "
        "Judge visual and semantic identity only. Do not infer or prefer a candidate "
        "because it may be closer. Assess every displayed Snapshot Object and Frontier."
    )
    if content:
        content[-1] = (
            "Return JSON as either an object with key assessments or directly as the assessments array. "
            "The assessments value must be a list "
            "containing each displayed candidate exactly once. Each item has "
            "candidate_id, confidence, rank_group, and evidence. candidate_id must be "
            "exactly 'Snapshot i, Object j' or 'Frontier i'. confidence must be high, "
            "medium, low, or no_match. rank_group is a nonnegative integer; use the same "
            "group only when candidates are semantically equivalent. For Snapshot objects, "
            "high means the visible object itself matches the requested identity; medium "
            "means it is useful evidence but requires a fresh view. For Frontiers, confidence "
            "means likelihood of revealing useful evidence. Confidence and evidence must be "
            "logically consistent: if the evidence says an object is not the requested target, "
            "its confidence must be low or no_match, never high. Keep evidence concise.\n",
        )
    candidate_manifest = "\n".join(f"- {candidate_id}" for candidate_id in candidate_ids)
    content.append((
        "Candidate manifest (copy candidate_id values exactly, including commas):\n"
        f"{candidate_manifest}\n"
        f"The assessments array must contain exactly {len(candidate_ids)} rows. "
        "An empty or partial array is invalid. Example row: "
        '{"candidate_id":"Snapshot 0, Object 0","confidence":"high",'
        '"rank_group":0,"evidence":"visible identity match"}.\n',
    ))

    parsed = None
    retry_bound = int(navigator.config.get("semantic_parse_attempts", 2))
    attempts = 0
    last_response = None
    for _ in range(max(1, retry_bound)):
        attempts += 1
        try:
            last_response = call_openai_api(
                sys_prompt,
                content,
                model_name=model_name,
                extra_body=extra_body,
                purpose="hierarchical_semantic_selection",
                max_tries=1,
            )
        except NonRetryableVLMError:
            break
        parsed = parse_semantic_assessments(last_response, candidate_ids)
        if parsed is not None:
            break
    navigator.stats["semantic_requests"] += 1
    fallback_used = parsed is None
    if parsed is None:
        logging.warning(
            "Hierarchical semantic protocol failed after %d attempt(s) for %d candidates; "
            "using Frontier-only fallback. Response excerpt: %r",
            attempts,
            len(candidate_ids),
            None if last_response is None else last_response[:600],
        )
        parsed = {}
    # Omitted objects can never cross the identity gate. Omitted Frontiers are
    # retained as low-confidence safe exploration fallbacks.
    for candidate_id in candidate_ids:
        if candidate_id not in parsed:
            parsed[candidate_id] = SemanticAssessment(
                candidate_id,
                SemanticConfidence.NO_MATCH if candidate_kinds[candidate_id] == "snapshot" else SemanticConfidence.LOW,
                10**6,
                "safe_default_after_missing_or_invalid_assessment",
            )
    confidence_counts = {}
    for assessment in parsed.values():
        label = assessment.confidence.value
        confidence_counts[label] = confidence_counts.get(label, 0) + 1
    navigator.last_semantic_protocol = {
        "candidate_count": len(candidate_ids),
        "parse_attempts": attempts,
        "fallback_used": fallback_used,
        "cache_hit": False,
        "confidence_counts": confidence_counts,
        "high_candidates": [
            key for key, value in parsed.items()
            if value.confidence == SemanticConfidence.HIGH
        ],
        "medium_candidates": [
            key for key, value in parsed.items()
            if value.confidence == SemanticConfidence.MEDIUM
        ],
    }
    navigator.store_assessments(digest, parsed)
    return parsed, snapshot_id_mapping, snapshot_crop_mapping, len(snapshot_full_imgs), digest
