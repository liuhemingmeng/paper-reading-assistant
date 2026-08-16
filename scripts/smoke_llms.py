"""Live smoke test for the candidate LLM chat models.

Sends one short Chinese academic prompt to each candidate and records
availability, latency, and a response snippet. API keys are read from the
environment (never hardcoded); the JSON report contains no secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import httpx

# Candidate list: public base URLs + user-provided model names. API keys are
# supplied via environment variables at run time (VOLCANO_LLM_API_KEY /
# SILICONFLOW_LLM_API_KEY) so secrets never touch this file or the report.
CANDIDATES = [
    {"name": "volcano-glm-4-7", "provider": "volcano", "base_url": "https://ark.cn-beijing.volces.com/api/v3", "api_key_env": "VOLCANO_LLM_API_KEY", "model": "glm-4-7-251222"},
    {"name": "volcano-doubao-seed-2-0-lite", "provider": "volcano", "base_url": "https://ark.cn-beijing.volces.com/api/v3", "api_key_env": "VOLCANO_LLM_API_KEY", "model": "doubao-seed-2-0-lite-260428"},
    {"name": "volcano-deepseek-v4-flash", "provider": "volcano", "base_url": "https://ark.cn-beijing.volces.com/api/v3", "api_key_env": "VOLCANO_LLM_API_KEY", "model": "deepseek-v4-flash-260425"},
    {"name": "siliconflow-deepseek-v3.2", "provider": "siliconflow", "base_url": "https://api.siliconflow.cn/v1", "api_key_env": "SILICONFLOW_LLM_API_KEY", "model": "deepseek-ai/DeepSeek-V3.2"},
    {"name": "siliconflow-qwen3.5-35b", "provider": "siliconflow", "base_url": "https://api.siliconflow.cn/v1", "api_key_env": "SILICONFLOW_LLM_API_KEY", "model": "Qwen/Qwen3.5-35B-A3B"},
]

PROMPT = (
    "用两句话向一名计算机本科生解释：什么是检索增强生成（RAG），"
    "以及它为什么能减少大模型的幻觉？只使用你已知的知识回答。"
)


def smoke_one(spec: dict) -> dict:
    key = os.getenv(spec["api_key_env"], "").strip()
    if not key:
        return {"name": spec["name"], "model": spec["model"], "provider": spec["provider"], "status": "no_key", "ok": False, "error": f"missing {spec['api_key_env']}"}
    payload = {
        "model": spec["model"],
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "你是一名严谨的中文机器学习助教。"},
            {"role": "user", "content": PROMPT},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    start = time.time()
    try:
        with httpx.Client(timeout=httpx.Timeout(40.0, connect=10.0)) as client:
            response = client.post(f"{spec['base_url']}/chat/completions", headers=headers, json=payload)
        latency_ms = round((time.time() - start) * 1000)
        if response.status_code >= 400:
            body = response.text[:200]
            return {"name": spec["name"], "model": spec["model"], "provider": spec["provider"], "status": "http_error", "ok": False, "latency_ms": latency_ms, "error": f"HTTP {response.status_code}: {body}"}
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "name": spec["name"],
            "model": spec["model"],
            "provider": spec["provider"],
            "status": "ok",
            "ok": True,
            "latency_ms": latency_ms,
            "sample": content[:200],
            "usage": usage,
        }
    except Exception as error:  # noqa: BLE001 - report any failure without leaking the key
        latency_ms = round((time.time() - start) * 1000)
        detail = str(error)[:200]
        return {"name": spec["name"], "model": spec["model"], "provider": spec["provider"], "status": "error", "ok": False, "latency_ms": latency_ms, "error": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/llm_smoke.json")
    args = parser.parse_args()

    results = [smoke_one(spec) for spec in CANDIDATES]
    report = {"prompt": PROMPT, "models": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    for record in results:
        flag = "OK  " if record["ok"] else "FAIL"
        print(f"[{flag}] {record['name']:30} {record.get('status'):12} lat={record.get('latency_ms', '-')}ms")
        if record["ok"]:
            print(f"        sample: {record['sample'][:80]}")
        else:
            print(f"        {record.get('error', '')[:120]}")
    ok_count = sum(1 for record in results if record["ok"])
    print(f"\n{ok_count}/{len(results)} LLM models reachable")


if __name__ == "__main__":
    main()
