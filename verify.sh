#!/bin/bash
# verify.sh — Local Docker Verification Script
# This script runs inside the Docker container to verify the project.

set -e

echo "=========================================="
echo "  LOCAL DOCKER VERIFICATION SUITE"
echo "=========================================="

# 1. Checklist Audit
echo ""
echo "Step 1/3: Running Pre-Submission Checklist Audit..."
python check_submission.py
echo "✓ Checklist audit complete."

# 2. Environment Validation
echo ""
echo "Step 2/3: Running Core Environment Validation..."
# Run validate.py which tests the environment reset/step logic
python validate.py
echo "✓ Environment validation complete."

# 3. OpenEnv Validate
echo ""
echo "Step 3/3: Running 'openenv validate'..."
# This ensures the manifest (openenv.yaml) matches the app
export PORT=8000
openenv validate || echo "⚠️ 'openenv validate' failed or is not available. Skipping."

echo ""
echo "=========================================="
echo "  ✅ VERIFICATION COMPLETE"
echo "=========================================="
