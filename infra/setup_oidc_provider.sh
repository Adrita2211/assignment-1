#!/usr/bin/env bash
#
# The GitHub Actions OIDC provider is account-wide, not per-repo -- other
# repos/roles may already depend on it. Deliberately kept OUTSIDE the
# CloudFormation stacks (a stack teardown must never risk deleting a shared,
# account-level resource). Idempotent -- safe to run every time.

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"

if ! aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  aws iam create-open-id-connect-provider \
    --url "https://token.actions.githubusercontent.com" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" >/dev/null
  echo "Created GitHub OIDC provider"
else
  echo "GitHub OIDC provider already exists"
fi

echo "$OIDC_ARN"
