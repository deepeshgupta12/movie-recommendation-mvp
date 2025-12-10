from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple  # noqa: UP035

import torch

from src.config.settings import settings

# ---------- Paths ----------
REPORTS_DIR = Path(getattr(settings, "REPORTS_DIR", "reports"))
MODELS_DIR = REPORTS_DIR / "models"


# ---------- Helpers ----------
def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _is_module(x: Any) -> bool:
    return isinstance(x, torch.nn.Module)


def _safe_state_dict(x: Any) -> Optional[Dict]:
    try:
        if _is_module(x):
            return x.state_dict()
    except Exception:
        return None
    return None


def _extract_prefixed(sd: Dict, prefix: str) -> Dict:
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _extract_from_dict(d: dict) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}

    # 1) Explicit keys
    for k in [
        "user_tower_state_dict",
        "item_tower_state_dict",
        "user_embedding_state_dict",
        "item_embedding_state_dict",
        "state_dict",
        "model_state_dict",
    ]:
        if isinstance(d.get(k), dict):
            out[k] = d[k]

    # 2) Embedded modules under common names
    for k in [
        "user_tower", "item_tower",
        "user_model", "item_model",
        "user_encoder", "item_encoder",
        "user_emb", "item_emb",
        "model",
        "two_tower",
    ]:
        v = d.get(k)
        sd = _safe_state_dict(v)
        if sd is not None:
            out[f"{k}_state_dict"] = sd

    # 3) Generic state_dict split by prefix
    generic = d.get("state_dict") or d.get("model_state_dict")
    if isinstance(generic, dict):
        user_pref = _extract_prefixed(generic, "user_tower.")
        item_pref = _extract_prefixed(generic, "item_tower.")
        if user_pref:
            out.setdefault("user_tower_state_dict", user_pref)
        if item_pref:
            out.setdefault("item_tower_state_dict", item_pref)

    # 4) Nested bags
    for nested_key in ["artifacts", "bundle", "payload", "checkpoint", "model_obj"]:
        nested = d.get(nested_key)
        if isinstance(nested, dict):
            out.update(_extract_from_dict(nested))

    return out


def _scan_object_for_modules(obj: Any) -> Dict[str, torch.nn.Module]:
    """
    Aggressively scan attributes for torch modules.
    We don't assume class structure.
    """
    found: Dict[str, torch.nn.Module] = {}

    # Try __dict__ first
    raw = getattr(obj, "__dict__", None)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if _is_module(v):
                found[k] = v

    # Also scan getattr list safely
    for name in dir(obj):
        if name.startswith("__"):
            continue
        try:
            v = getattr(obj, name)
        except Exception:
            continue
        if _is_module(v):
            found.setdefault(name, v)

    return found


def _best_user_item_modules(mods: Dict[str, torch.nn.Module]) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    """
    Heuristics to identify user tower, item tower, or a parent two-tower model.
    """
    user = None
    item = None
    parent = None

    # Prioritize explicit-looking names
    for k, m in mods.items():
        lk = k.lower()
        if user is None and any(s in lk for s in ["user_tower", "user_encoder", "user_model", "user_net"]):
            user = m
        if item is None and any(s in lk for s in ["item_tower", "item_encoder", "item_model", "item_net", "movie_tower"]):
            item = m

    # Embedding-only fallbacks
    for k, m in mods.items():
        lk = k.lower()
        if user is None and any(s in lk for s in ["user_emb", "user_embedding", "u_emb"]):
            user = m
        if item is None and any(s in lk for s in ["item_emb", "item_embedding", "i_emb", "movie_emb"]):
            item = m

    # If we still don't have both, pick a "main model"
    # based on name hints
    for k, m in mods.items():
        lk = k.lower()
        if any(s in lk for s in ["two_tower", "model", "net", "recommender"]):
            parent = m
            break

    # As last resort, if exactly 1 module exists, treat as parent
    if parent is None and len(mods) == 1:
        parent = next(iter(mods.values()))

    return user, item, parent


def _extract_from_object(obj: Any) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}

    # 1) Direct attribute dicts
    for attr in [
        "user_tower_state_dict",
        "item_tower_state_dict",
        "user_embedding_state_dict",
        "item_embedding_state_dict",
        "state_dict",
        "model_state_dict",
    ]:
        v = getattr(obj, attr, None)
        if isinstance(v, dict):
            out[attr] = v

    # 2) If object is itself a Module
    if _is_module(obj):
        out.setdefault("state_dict", obj.state_dict())
        return out

    # 3) Pydantic dumps
    for fn_name in ["model_dump", "dict"]:
        fn = getattr(obj, fn_name, None)
        if callable(fn):
            try:
                dumped = fn()
                if isinstance(dumped, dict):
                    out.update(_extract_from_dict(dumped))
            except Exception:
                pass

    # 4) Raw __dict__
    raw = getattr(obj, "__dict__", None)
    if isinstance(raw, dict):
        out.update(_extract_from_dict(raw))

    # 5) Aggressive module scan
    mods = _scan_object_for_modules(obj)
    user_m, item_m, parent_m = _best_user_item_modules(mods)

    if user_m is not None:
        out.setdefault("user_tower_state_dict", user_m.state_dict())
    if item_m is not None:
        out.setdefault("item_tower_state_dict", item_m.state_dict())

    # If still missing tower splits but we have a parent, store generic state_dict
    if parent_m is not None:
        out.setdefault("state_dict", parent_m.state_dict())

        # Try prefix split from parent
        user_pref = _extract_prefixed(out["state_dict"], "user_tower.")
        item_pref = _extract_prefixed(out["state_dict"], "item_tower.")
        if user_pref:
            out.setdefault("user_tower_state_dict", user_pref)
        if item_pref:
            out.setdefault("item_tower_state_dict", item_pref)

    return out


def main():
    pt_path = MODELS_DIR / "two_tower_v3.pt"
    meta_path = MODELS_DIR / "two_tower_v3.meta.json"

    if not pt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {pt_path}")

    print("\n[START] Normalizing Two-Tower V3 checkpoint (aggressive)...")
    obj = torch.load(pt_path, map_location="cpu")

    extracted: Dict[str, Dict] = {}

    if isinstance(obj, dict):
        extracted.update(_extract_from_dict(obj))
    else:
        extracted.update(_extract_from_object(obj))

    # Accept either tower splits OR a usable generic state_dict
    has_user = any(k in extracted for k in ["user_tower_state_dict", "user_embedding_state_dict"])
    has_item = any(k in extracted for k in ["item_tower_state_dict", "item_embedding_state_dict"])
    has_generic = any(k in extracted for k in ["state_dict", "model_state_dict"])

    if not ((has_user and has_item) or has_generic):
        raise RuntimeError(
            "Could not extract tower or model weights from existing checkpoint. "
            "Your two_tower_v3.pt appears to be saved in a structure that does not retain "
            "torch modules/state_dicts. In that case we must re-save via training script."
        )

    meta = _load_meta(meta_path)

    compat = {
        "format": "two_tower_v3_compat",
        "meta": meta,
        **extracted,
    }

    torch.save(compat, pt_path)

    print("[DONE] Checkpoint normalized and overwritten in compatible format.")
    print(f"[PATH] {pt_path.resolve()}")
    print("[OK] Keys saved:")
    for k in sorted(compat.keys()):
        if k != "meta":
            print(f" - {k}")


if __name__ == "__main__":
    main()