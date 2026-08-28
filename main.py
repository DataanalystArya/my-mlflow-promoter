from datetime import datetime
import math
import re
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="MLflow Model Promotion Gate")


def parse_iso(ts_str: Any) -> Optional[datetime]:
  if not isinstance(ts_str, str):
    return None
  # Regex check for YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)
  pattern = (
      r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$"
  )
  if not re.match(pattern, ts_str):
    return None
  try:
    s = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(s)
  except Exception:
    return None


def is_canonical_pos_int_str(s: Any) -> bool:
  if not isinstance(s, str):
    return False
  return bool(re.match(r"^[1-9]\d*$", s))


def is_finite_number(val: Any) -> bool:
  if val is None or isinstance(val, bool):
    return False
  if not isinstance(val, (int, float)):
    return False
  return not (math.isnan(val) or math.isinf(val))


@app.post("/promote")
async def promote(request: Request):
  try:
    body = await request.json()
  except Exception:
    return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

  if not isinstance(body, dict):
    return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

  as_of_str = body.get("asOf")
  champion_version = body.get("championVersion")
  policy = body.get("policy")
  versions = body.get("versions")

  # Validate high-level inputs
  if (
      not isinstance(as_of_str, str)
      or not is_canonical_pos_int_str(champion_version)
      or not isinstance(policy, dict)
      or not isinstance(versions, list)
  ):
    return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

  as_of_dt = parse_iso(as_of_str)
  if as_of_dt is None:
    return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

  # Extract policy fields
  req_dataset_digest = policy.get("datasetDigest")
  req_schema_digest = policy.get("schemaDigest")
  max_age_sec = policy.get("maxAgeSeconds")
  acc_floor = policy.get("accuracyFloor")
  req_slices = policy.get("requiredSlices", {})
  max_lat = policy.get("maxLatencyMs")
  max_size = policy.get("maxSizeBytes")
  min_imp = policy.get("minImprovement", 0.01)

  # Validate policy requirements
  if not (
      isinstance(req_dataset_digest, str)
      and len(req_dataset_digest) > 0
      and isinstance(req_schema_digest, str)
      and len(req_schema_digest) > 0
      and isinstance(max_age_sec, int)
      and max_age_sec >= 0
      and is_finite_number(acc_floor)
      and 0.0 <= acc_floor <= 1.0
      and isinstance(req_slices, dict)
      and is_finite_number(max_lat)
      and max_lat >= 0
      and isinstance(max_size, int)
      and max_size >= 0
      and is_finite_number(min_imp)
  ):
    return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

  failed_gates: Dict[str, List[str]] = {}
  seen_versions = set()
  eligible_versions = []
  version_eval_map = {}

  for v_entry in versions:
    if not isinstance(v_entry, dict):
      return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    v_id = v_entry.get("version")
    v_gates = set()

    # Check canonical version
    if not is_canonical_pos_int_str(v_id):
      v_gates.add("INVALID_VERSION")
      key = str(v_id) if v_id is not None else "unknown"
      failed_gates[key] = sorted(list(v_gates))
      continue

    # Check duplicate version
    if v_id in seen_versions:
      v_gates.add("DUPLICATE_VERSION")
      failed_gates[v_id] = sorted(list(v_gates))
      continue

    seen_versions.add(v_id)
    art_digest = v_entry.get("artifactDigest")
    eval_obj = v_entry.get("evaluation")

    if not isinstance(eval_obj, dict):
      v_gates.add("MISSING_EVALUATION")
      failed_gates[v_id] = sorted(list(v_gates))
      continue

    # Timestamp checks
    created_at_str = eval_obj.get("createdAt")
    created_at_dt = parse_iso(created_at_str)

    if created_at_dt is None:
      v_gates.add("INVALID_TIMESTAMP")
    else:
      diff_sec = (as_of_dt - created_at_dt).total_seconds()
      if diff_sec < 0:
        v_gates.add("FUTURE_EVALUATION")
      elif diff_sec > max_age_sec:
        v_gates.add("STALE_EVALUATION")

    # Digest mismatch checks
    if eval_obj.get("artifactDigest") != art_digest:
      v_gates.add("ARTIFACT_MISMATCH")
    if eval_obj.get("datasetDigest") != req_dataset_digest:
      v_gates.add("DATASET_MISMATCH")
    if eval_obj.get("schemaDigest") != req_schema_digest:
      v_gates.add("SCHEMA_MISMATCH")

    # Aggregate metric checks
    acc = eval_obj.get("accuracy")
    lat = eval_obj.get("latencyMs")
    sz = eval_obj.get("sizeBytes")

    if (
        not is_finite_number(acc)
        or not is_finite_number(lat)
        or not is_finite_number(sz)
    ):
      v_gates.add("NON_FINITE")
    else:
      if not (0.0 <= acc <= 1.0):
        v_gates.add("METRIC_RANGE")
      elif acc < acc_floor:
        v_gates.add("ACCURACY_FLOOR")

      if lat < 0 or lat > max_lat:
        v_gates.add("LATENCY_LIMIT")

      if sz < 0 or sz > max_size:
        v_gates.add("SIZE_LIMIT")

    # Slices check
    slices = eval_obj.get("slices")
    if not isinstance(slices, dict):
      v_gates.add("NON_FINITE")
    else:
      for req_s_name, req_s_floor in req_slices.items():
        if req_s_name not in slices:
          v_gates.add(f"MISSING_SLICE:{req_s_name}")
        else:
          s_val = slices[req_s_name]
          if not is_finite_number(s_val) or not (0.0 <= s_val <= 1.0):
            v_gates.add(f"SLICE_RANGE:{req_s_name}")
          elif s_val < req_s_floor:
            v_gates.add(f"SLICE_FLOOR:{req_s_name}")

    if v_gates:
      failed_gates[v_id] = sorted(list(v_gates))
    else:
      eligible_versions.append(v_id)
      version_eval_map[v_id] = eval_obj

  # Champion validity check
  if champion_version not in eligible_versions:
    return {
        "action": "block",
        "championVersion": champion_version,
        "selectedVersion": None,
        "eligibleVersions": eligible_versions,
        "failedGates": failed_gates,
        "aliasMutation": None,
        "evidence": None,
    }

  # Ranking eligible versions: Accuracy DESC, Latency ASC, Size ASC, Version numeric ASC
  def sort_key(v_id):
    ev = version_eval_map[v_id]
    return (-ev["accuracy"], ev["latencyMs"], ev["sizeBytes"], int(v_id))

  ranked = sorted(eligible_versions, key=sort_key)
  challenger = ranked[0]

  if challenger == champion_version:
    return {
        "action": "retain",
        "championVersion": champion_version,
        "selectedVersion": champion_version,
        "eligibleVersions": eligible_versions,
        "failedGates": failed_gates,
        "aliasMutation": None,
        "evidence": version_eval_map[champion_version],
    }

  champ_acc = version_eval_map[champion_version]["accuracy"]
  challenger_acc = version_eval_map[challenger]["accuracy"]
  improvement = round(challenger_acc - champ_acc, 12)

  if improvement >= min_imp:
    return {
        "action": "promote",
        "championVersion": champion_version,
        "selectedVersion": challenger,
        "eligibleVersions": eligible_versions,
        "failedGates": failed_gates,
        "aliasMutation": {"alias": "champion", "version": challenger},
        "evidence": version_eval_map[challenger],
    }

  return {
      "action": "retain",
      "championVersion": champion_version,
      "selectedVersion": champion_version,
      "eligibleVersions": eligible_versions,
      "failedGates": failed_gates,
      "aliasMutation": None,
      "evidence": version_eval_map[champion_version],
  }


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)
