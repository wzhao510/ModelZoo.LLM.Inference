


def cal_gpu_count(model_cfg: dict):
    serve_cfg = model_cfg.get("serve_config", {})
    tp = int(serve_cfg.get("tp", 1))
    pp = int(serve_cfg.get("pp", 1))
    dp = int(serve_cfg.get("dp", 1))
    return tp * pp * dp