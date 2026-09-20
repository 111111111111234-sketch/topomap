"""Goal verification is separate from HGR room/hypothesis verification."""

import numpy as np

from src.eval_utils_gpt_goatbench import encode_tensor2base64
from src.query_place_goal import _goal_content, validated_request


def verify_entity(metadata, current_rgb, choice, cfg, purpose="dual_topo_goal_verification"):
    crop = getattr(choice, "hierarchical_candidate_crop", None)
    if crop is None:
        return "error", {"reason": "selected_entity_evidence_missing"}
    try:
        goal_content = _goal_content(metadata)
    except (OSError, TypeError, ValueError):
        return "error", {"reason": "goal_evidence_unavailable"}
    content = goal_content + [
        ("Selected physical entity, historical evidence:", encode_tensor2base64(np.asarray(crop))),
        ("Fresh terminal observation:", encode_tensor2base64(np.asarray(current_rgb))),
    ]
    prompt = (
        'Verify this selected physical entity against the goal. Return JSON '
        '{"verdict":"matched|rejected|uncertain","reason":"visual evidence"}. '
        'matched requires the SAME physical entity visible now and matching the goal, '
        'including instance details for image/description goals. A nearby different '
        'object cannot confirm the selected entity. rejected requires explicit visible '
        'contradiction about the selected entity. Occlusion, absence, missing association '
        'or ambiguity means uncertain. Never reject a whole place.'
    )
    from omegaconf import OmegaConf
    request_cfg = OmegaConf.merge(cfg, {"place_goal_navigation": {"evidence_coverage": True}})
    result, info = validated_request(prompt, content, request_cfg, purpose)
    return ("error", info) if result is None else (
        {"matched": "confirmed", "rejected": "rejected", "uncertain": "uncertain"}[result["verdict"]],
        {**result, **info})
