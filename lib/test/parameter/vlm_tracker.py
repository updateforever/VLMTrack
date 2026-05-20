"""
Parameters for the unified VLM tracker.

Examples:
    api_mosaic
    api_pair
    api_mosaic_ref
    api_dense_mosaic
    vllm_qwen25_vl_7b_mosaic
    local_qwen3vl_4b_thinking_mosaic
"""
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.test.parameter.vlm_common import apply_vlm_config


_CONTEXT_TOKENS = {"pair", "mosaic"}


def parameters(tracker_param: str = "api_mosaic"):
    params = TrackerParams()
    env = env_settings()

    model_token, extras = _parse_tracker_param(tracker_param)
    apply_vlm_config(params, model_token)

    params.context_mode = extras["context_mode"]
    params.use_keyframe = extras["run_mode"] == "sparse"
    params.run_mode = extras["run_mode"]
    params.use_init_anchor = True
    params.use_init_bbox_ref = extras["use_init_bbox_ref"]
    params.history_policy = extras["history_policy"]
    params.history_buffer_size = extras["history_buffer_size"]
    params.prompt_name = "cognitivebench"
    params.vlm_max_image_side = extras["vlm_max_image_side"]
    params.output_reasoning = extras["output_reasoning"]
    params.target_text_history_size = extras["target_text_history_size"]

    params.temperature = extras["temperature"]
    params.max_new_tokens = extras["max_new_tokens"]
    params.debug = extras["debug"]
    params.save_all_boxes = False

    # Kept for legacy datasets. CognitiveBench carries keyframes in the dataset.
    params.keyframe_root = getattr(env, 'keyframe_root', '')
    params.checkpoint = None

    return params


def _parse_tracker_param(tracker_param: str):
    tokens = [t for t in (tracker_param or "api_mosaic").lower().split("_") if t]

    extras = {
        "run_mode": "sparse",
        "context_mode": "mosaic",
        "use_init_bbox_ref": False,
        "history_policy": "visible_keyframes",
        "history_buffer_size": 3,
        "vlm_max_image_side": 648,
        "output_reasoning": True,
        "target_text_history_size": 3,
        "temperature": 0.1,
        "max_new_tokens": 512,
        "debug": 0,
    }

    remaining = []
    for token in tokens:
        if token == "dense":
            extras["run_mode"] = "dense"
        elif token == "sparse":
            extras["run_mode"] = "sparse"
        elif token in _CONTEXT_TOKENS:
            extras["context_mode"] = token
        elif token == "ref":
            extras["use_init_bbox_ref"] = True
        elif token == "fast":
            extras["output_reasoning"] = False
        elif token == "slow":
            extras["output_reasoning"] = True
        elif token == "all":
            extras["history_policy"] = "all_keyframes"
        elif token == "none":
            extras["history_policy"] = "none"
        elif token.startswith("b") and token[1:].isdigit():
            extras["history_buffer_size"] = int(token[1:])
        elif token.startswith("text") and token[4:].isdigit():
            extras["target_text_history_size"] = int(token[4:])
        elif token.startswith("side") and token[4:].isdigit():
            extras["vlm_max_image_side"] = int(token[4:])
        elif token.startswith("tok") and token[3:].isdigit():
            extras["max_new_tokens"] = int(token[3:])
        elif token.startswith("temp"):
            try:
                extras["temperature"] = float(token[4:].replace("p", "."))
            except ValueError:
                remaining.append(token)
        elif token.startswith("dbg") and token[3:].isdigit():
            extras["debug"] = int(token[3:])
        else:
            remaining.append(token)

    model_token = "_".join(remaining) if remaining else "api"
    if model_token == "vllm":
        raise ValueError("vllm model alias is incomplete, e.g. vllm_qwen25_vl_7b_mosaic")
    if extras["run_mode"] == "dense" and extras["history_policy"] == "visible_keyframes":
        extras["history_policy"] = "visible_all"

    return model_token, extras
