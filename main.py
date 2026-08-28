from datetime import datetime
import math
import os
import re
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="MLflow Model Promotion Gate")


def parse_iso(ts_str: Any) -> Optional[datetime]:
  if not isinstance(ts_str, str):
    return None
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
  if not re.match(r"^[1-9]\d*$", s):
    return False
  try:
    val = int(s)
    return val <= 9007199254740991
  except Exception:
    return False


def is_finite_number(val: Any) -> bool:
  if val is None or isinstance(val, bool):
    return False
  if not isinstance(val, (int, float)):
    return False
  return math.isfinite(val)


def is_non_neg_int(val: Any) -> bool:
  if val is None or isinstance(val, bool):
    return False
  if not isinstance(val, int):
    return False
  return 0 <= val <= 9007199254740991


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

  # Mandatory HTTP 400 validations
  if (
      not isinstance(as_of_str, str)
      or parse_iso(as_of_str) is None
      or not isinstance(champion_version, str)
      or not isinstance(policy, dict)
      or not isinstance(versions, list)
  ):
    return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

  as_of_dt = parse_iso(as_of_str)

  # Check policy integrity
  req_dataset_digest = policy.get("datasetDigest")
  req_schema_digest = policy.get("schemaDigest")
  max_age_sec = policy.get("maxAgeSeconds")
  acc_floor = policy.get("accuracyFloor")
  req_slices = policy.get("requiredSlices")
  max_lat = policy.get("maxLatencyMs")
  max_size = policy.get("maxSizeBytes")
  min_imp = policy.get("minImprovement", 0.01)

  policy_valid = True
  if not (
      isinstance(req_dataset_digest, str)
      and len(req_dataset_digest) > 0
      and isinstance(req_schema_digest, str)
      and len(req_schema_digest) > 0
      and is_non_neg_int(max_age_sec)
      and is_finite_number(acc_floor)
      and 0.0 <= acc_floor <= 1.0
      and isinstance(req_slices, dict)
      and is_finite_number(max_lat)
      and max_lat >= 0
      and is_non_neg_int(max_size)
      and is_finite_number(min_imp)
      and 0.0 <= min_imp <= 1.0
  ):
    policy_valid = False

  if policy_valid:
    for sk, sv in req_slices.items():
      if (
          not isinstance(sk, str)
          or not is_finite_number(sv)
          or not (0.0 <= sv <= 1.0)
      ):
        policy_valid = False
        break

  # Pre-scan duplicate versions
  version_counts: Dict[str, int] = {}
  for v_entry in versions:
    if isinstance(v_entry, dict):
      vid = v_entry.get("version")
      if isinstance(vid, str):
        version_counts[vid] = version_counts.get(vid, 0) + 1

  failed_gates: Dict[str, List[str]] = {}
  eligible_versions: List[str] = []
  version_eval_map: Dict[str, dict] = {}

  for v_entry in versions:
    if not isinstance(v_entry, dict):
      return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    v_id = v_entry.get("version")
    v_gates = set()
    v_key = str(v_id) if v_id is not None else "unknown"

    if not is_canonical_pos_int_str(v_id):
      v_gates.add("INVALID_VERSION")

    if isinstance(v_id, str) and version_counts.get(v_id, 0) > 1:
      v_gates.add("DUPLICATE_VERSION")

    if not policy_valid:
      v_gates.add("INVALID_POLICY")

    eval_obj = v_entry.get("evaluation")
    if not isinstance(eval_obj, dict):
      v_gates.add("MISSING_EVALUATION")
      failed_gates[v_key] = sorted(list(v_gates))
      continue

    # Timestamps
    created_at_str = eval_obj.get("createdAt")
    created_at_dt = parse_iso(created_at_str)
    if created_at_dt is None:
      v_gates.add("INVALID_TIMESTAMP")
    else:
      diff_sec = (as_of_dt - created_at_dt).total_seconds()
      if diff_sec < 0:
        v_gates.add("FUTURE_EVALUATION")
      elif policy_valid and diff_sec > max_age_sec:
        v_gates.add("STALE_EVALUATION")

    # Digests
    art_digest = v_entry.get("artifactDigest")
    if (
        eval_obj.get("artifactDigest") != art_digest
        or not isinstance(art_digest, str)
        or len(art_digest) == 0
    ):
      v_gates.add("ARTIFACT_MISMATCH")
    if policy_valid:
      if eval_obj.get("datasetDigest") != req_dataset_digest:
        v_gates.add("DATASET_MISMATCH")
      if eval_obj.get("schemaDigest") != req_schema_digest:
        v_gates.add("SCHEMA_MISMATCH")

    # Aggregate Metrics
    acc = eval_obj.get("accuracy")
    lat = eval_obj.get("latencyMs")
    sz = eval_obj.get("sizeBytes")

    acc_finite = is_finite_number(acc)
    lat_finite = is_finite_number(lat)
    sz_finite = is_non_neg_int(sz)

    if not (acc_finite and lat_finite and sz_finite):
      v_gates.add("NON_FINITE")
    else:
      if not (0.0 <= acc <= 1.0):
        v_gates.add("METRIC_RANGE")
      elif policy_valid and acc < acc_floor:
        v_gates.add("ACCURACY_FLOOR")

      if lat < 0:
        v_gates.add("METRIC_RANGE")
      elif policy_valid and lat > max_lat:
        v_gates.add("LATENCY_LIMIT")

      if sz < 0:
        v_gates.add("METRIC_RANGE")
      elif policy_valid and sz > max_size:
        v_gates.add("SIZE_LIMIT")

    # Slices
    slices = eval_obj.get("slices")
    if not isinstance(slices, dict):
      v_gates.add("NON_FINITE")
    elif policy_valid:
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
      failed_gates[v_key] = sorted(list(set(v_gates)))
    else:
      eligible_versions.append(v_id)
      version_eval_map[v_id] = eval_obj

  # Ranking: accuracy desc, latency asc, size asc, version asc
  def sort_key(vid: str):
    ev = version_eval_map[vid]
    return (-ev["accuracy"], ev["latencyMs"], ev["sizeBytes"], int(vid))

  ranked_eligible = sorted(eligible_versions, key=sort_key)

  if champion_version not in eligible_versions:
    return {
        "action": "block",
        "championVersion": champion_version,
        "selectedVersion": None,
        "eligibleVersions": ranked_eligible,
        "failedGates": failed_gates,
        "aliasMutation": None,
        "evidence": None,
    }

  challenger = ranked_eligible[0]

  if challenger == champion_version:
    return {
        "action": "retain",
        "championVersion": champion_version,
        "selectedVersion": champion_version,
        "eligibleVersions": ranked_eligible,
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
        "eligibleVersions": ranked_eligible,
        "failedGates": failed_gates,
        "aliasMutation": {"alias": "champion", "version": challenger},
        "evidence": version_eval_map[challenger],
    }

  return {
      "action": "retain",
      "championVersion": champion_version,
      "selectedVersion": champion_version,
      "eligibleVersions": ranked_eligible,
      "failedGates": failed_gates,
      "aliasMutation": None,
      "evidence": version_eval_map[champion_version],
  }


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8000))
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=port)
