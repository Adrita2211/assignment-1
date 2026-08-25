#!/usr/bin/env bash
#
# Tears down everything infra/create_kb_aurora.sh provisions, in dependency
# order (Knowledge Base before the Aurora cluster it points at, instance
# before cluster). Does NOT delete the S3 bucket or IAM role by default --
# both are cheap to leave and useful to keep around if you're going to
# re-provision again; pass --full to remove those too.
#
# Required env vars:
#   None strictly required, but the identifiers below assume you used
#   create_kb_aurora.sh's defaults -- override if you didn't.
#
# Optional:
#   AWS_REGION  - defaults to us-east-1
#   APP_NAME    - defaults to ecommerce-agent
#   KB_ID       - Knowledge Base ID to delete (skips KB deletion if unset)

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-ecommerce-agent}"
CLUSTER_ID="${APP_NAME}-kb-aurora"
FULL="${1:-}"

if [ -n "${KB_ID:-}" ]; then
  echo "== Deleting Knowledge Base ${KB_ID} =="
  aws bedrock-agent delete-knowledge-base --knowledge-base-id "$KB_ID" --region "$AWS_REGION" || true
else
  echo "KB_ID not set -- skipping Knowledge Base deletion (delete it manually first if one exists,"
  echo "or the Aurora cluster delete below will leave an orphaned KB pointing at a dead cluster)."
fi

echo "== Deleting Aurora instance ${CLUSTER_ID}-instance-1 =="
aws rds delete-db-instance --db-instance-identifier "${CLUSTER_ID}-instance-1" --skip-final-snapshot --region "$AWS_REGION" || true
aws rds wait db-instance-deleted --db-instance-identifier "${CLUSTER_ID}-instance-1" --region "$AWS_REGION" || true

echo "== Deleting Aurora cluster ${CLUSTER_ID} =="
aws rds delete-db-cluster --db-cluster-identifier "$CLUSTER_ID" --skip-final-snapshot --region "$AWS_REGION"

if [ "$FULL" = "--full" ]; then
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
  BUCKET="${APP_NAME}-kb-policies-${ACCOUNT_ID}"
  ROLE_NAME="${APP_NAME}-kb-service-role"
  echo "== --full: removing S3 bucket and IAM role too =="
  aws s3 rb "s3://${BUCKET}" --force --region "$AWS_REGION" || true
  aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name kb-access-policy --region "$AWS_REGION" || true
  aws iam delete-role --role-name "$ROLE_NAME" --region "$AWS_REGION" || true
fi

echo "Done."
