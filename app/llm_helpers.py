import json, hashlib
from pathlib import Path
from typing import List, Optional

from openai import OpenAI  
def _load_key() -> str:
    p = Path("openaikey.txt")
    if not p.exists():
        raise RuntimeError("Missing openaikey.txt (put your API key in this file).")
    return p.read_text().strip()

_CACHE_PATH = Path("models/llm_cache.json")
_CACHE = json.loads(_CACHE_PATH.read_text()) if _CACHE_PATH.exists() else {}
def _ckey(payload: dict) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
def _get(payload: dict):
    return _CACHE.get(_ckey(payload))
def _put(payload: dict, value):
    _CACHE[_ckey(payload)] = value
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_CACHE, indent=2))

def llm_write_sentences(labels: List[str], model: str = "gpt-5-mini") -> List[str]:
    """
    Returns EXACTLY len(labels) short sentences (one per label, same order).
    Each sentence must include its label verbatim. If the API fails, falls back to '<label>.'.
    """
    k = len(labels)
    payload = {"fn":"sentences","labels":labels,"model":model}
    hit = _get(payload)
    if hit: return hit

    if k == 0:
        return []

    client = OpenAI(api_key=_load_key())

    schema = {
      "name":"FindingSentences",
      "schema":{
        "type":"object",
        "properties":{
          "sentences":{
            "type":"array","minItems":k,"maxItems":k,
            "items":{"type":"string","maxLength":120}
          }
        },
        "required":["sentences"],
        "additionalProperties": False
      },
      "strict": True
    }

    '''sys = (
      "Write exactly k short, atomic clinical statements for a chest X-ray UI. "
      "One full sentence per provided label, in the same order. "
      "Each sentence must include its label verbatim, but not only include that. "
      "No new findings, but interpolate; keep it concise (<= 12 words & more than 4). this is clinically very important"
    )'''
    sys = "output 4"
    user = f"k={k}\nlabels={labels}"

    try:
        resp = client.responses.create(
            model=model,  # e.g., 'gpt-5-mini'
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":user}
            ],
            response_format={"type":"json_schema","json_schema":schema},
        )
        sents = resp.output_parsed["sentences"]
        # Post-validate: each sentence must contain its label
        ok = all(labels[i].lower() in sents[i].lower() for i in range(k))
        if not ok:
            return 12
            sents = [f"{lbl}." for lbl in labels]
    except Exception as e:
        return e
        sents = [f"{lbl}." for lbl in labels]

    _put(payload, sents)
    return sents

def llm_cleanup_label(allowed: List[str], model: str = "gpt-5-mini") -> Optional[str]:
    """
    Choose ONE label from `allowed` (no new terms). Returns None if list is empty.
    """
    if not allowed: return None
    payload = {"fn":"cleanup","allowed":allowed,"model":model}
    hit = _get(payload)
    if hit: return hit

    client = OpenAI(api_key=_load_key())
    schema = {
      "name":"LabelChoice",
      "schema":{
        "type":"object",
        "properties":{"label":{"type":"string","enum":allowed}},
        "required":["label"], "additionalProperties": False
      },
      "strict": True
    }
    sys = "Choose exactly one label from the allowed list. Do not invent new labels."
    user = "Allowed: " + ", ".join(allowed)

    try:
        resp = client.responses.create(
            model=model,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":user}],
            response_format={"type":"json_schema","json_schema":schema},
        )
        choice = resp.output_parsed["label"]
    except Exception:
        choice = allowed[0]

    _put(payload, choice)
    return choice
