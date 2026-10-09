def observations(result):
    """Convert tensors once to native values (including optional tracking IDs)."""
    if result.boxes is None:
        return []
    boxes = result.boxes.cpu()
    coords = boxes.xyxy.tolist()
    classes = boxes.cls.tolist()
    confidence = boxes.conf.tolist()
    ids = boxes.id.tolist() if boxes.id is not None else [None] * len(coords)
    return [{"box": box, "class": int(cls), "label": result.names[int(cls)],
             "confidence": float(conf), "id": int(track_id) if track_id is not None else None}
            for box, cls, conf, track_id in zip(coords, classes, confidence, ids)]
