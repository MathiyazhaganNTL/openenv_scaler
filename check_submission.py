"""Pre-Submission Checklist Auditor — checks inference.py against all 5 rules."""
import re, sys

path = r"G:\CLG_Hacks\Hackathons\13.openenv\openenv\inference.py"
src = open(path, "r", encoding="utf-8").read()

checks = []

# 1. Read the sample inference.py and followed it strictly
#    (proxy: has the key structural elements)
c1 = ("from openai import OpenAI" in src
      and "def call_llm" in src
      and "def main" in src)
checks.append(("1. Follows sample inference.py structure", c1))

# 2. Environment variables present
has_api   = bool(re.search(r'API_BASE_URL\s*=\s*os\.getenv\(', src))
has_model = bool(re.search(r'MODEL_NAME\s*=\s*os\.getenv\(', src))
has_hf    = bool(re.search(r'HF_TOKEN\s*=\s*os\.getenv\(', src))
has_local = bool(re.search(r'LOCAL_IMAGE_NAME\s*=\s*os\.getenv\(', src))
c2 = has_api and has_model and has_hf and has_local
detail2 = f"API_BASE_URL={has_api}, MODEL_NAME={has_model}, HF_TOKEN={has_hf}, LOCAL_IMAGE_NAME={has_local}"
checks.append((f"2. All env vars present ({detail2})", c2))

# 3. Defaults set ONLY for API_BASE_URL and MODEL_NAME (not HF_TOKEN)
api_default   = bool(re.search(r'API_BASE_URL\s*=\s*os\.getenv\([^,]+,\s*"', src))
model_default = bool(re.search(r'MODEL_NAME\s*=\s*os\.getenv\([^,]+,\s*"', src))
# HF_TOKEN must NOT have a second argument (no default)
hf_no_default = bool(re.search(r'HF_TOKEN\s*=\s*os\.getenv\(\s*"HF_TOKEN"\s*\)', src))
c3 = api_default and model_default and hf_no_default
checks.append((f"3. Defaults: API_BASE_URL=Yes({api_default}), MODEL_NAME=Yes({model_default}), HF_TOKEN=No({hf_no_default})", c3))

# 4. All LLM calls use OpenAI client
c4 = "chat.completions.create" in src and "from openai import OpenAI" in src
checks.append(("4. LLM calls use OpenAI SDK (chat.completions.create)", c4))

# 5. Stdout logs follow START/STEP/END format
c5 = "[START]" in src and "[STEP]" in src and "[END]" in src
checks.append(("5. Stdout logs use [START]/[STEP]/[END] format", c5))

print()
print("=" * 60)
print("  Pre-Submission Checklist — Automated Audit")
print("=" * 60)
passed = 0
for name, ok in checks:
    mark = "PASS" if ok else "FAIL"
    sym  = "OK" if ok else "XX"
    print(f"  [{mark}] {sym}  {name}")
    if ok:
        passed += 1
print("-" * 60)
print(f"  RESULT: {passed}/{len(checks)} checks passed")
if passed == len(checks):
    print("  >> Your submission is READY!")
else:
    print("  >> Fix the failing checks before submitting.")
print("=" * 60)
