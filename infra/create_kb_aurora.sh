#!/usr/bin/env bash
#
# Provisions the Assignment 3 §2.3 Bedrock Knowledge Base retrieval stack:
# an Aurora Serverless v2 PostgreSQL cluster (with the RDS Data API enabled,
# so callers -- including AgentCore Runtime -- need no VPC networking to
# reach it), a pgvector schema matching Bedrock KB's required table shape,
# an S3 bucket holding policies/*.md, an IAM service role for the Knowledge
# Base, and the Knowledge Base + S3 data source + ingestion job themselves.
#
# This is the exact sequence used to build the real, live-verified stack
# this project's README documents -- not a from-scratch design, a script
# capturing what was actually run, so it's reproducible and so the real
# gotchas hit along the way (documented inline below) aren't lost.
#
# Prerequisites (not automated here, real one-time account setup):
#   - Bedrock model access granted for amazon.titan-embed-text-v2:0 in this
#     account/region (see README's "Bedrock model access" section for how
#     this project discovered the account needed a console Playground
#     invocation to trigger auto-enablement -- a bare API/CLI InvokeModel
#     call was NOT sufficient on this account).
#   - An existing RDS subnet group in the target VPC (this project reused
#     the one Assignment 2's CloudFormation stack already created).
#
# Required env vars:
#   DB_SUBNET_GROUP_NAME  - existing RDS subnet group to place the cluster in
#
# Optional:
#   AWS_REGION            - defaults to us-east-1
#   APP_NAME              - defaults to ecommerce-agent
#   AURORA_MIN_ACU        - defaults to 0.5 (Aurora Serverless v2 minimum)
#   AURORA_MAX_ACU        - defaults to 1 (kept small deliberately -- this
#                           project's policy corpus is 7 short markdown
#                           files; no real reason to provision headroom
#                           beyond what a demo/eval workload needs)
#
# NOT idempotent for the CREATE calls (unlike deploy_agent_stack.sh's
# CloudFormation-based idempotency) -- re-running this against
# already-existing resources will error on the create-db-cluster /
# create-knowledge-base calls. Tear down first (teardown_kb_aurora.sh) if
# re-running from scratch.

set -euo pipefail

: "${DB_SUBNET_GROUP_NAME:?set DB_SUBNET_GROUP_NAME to an existing RDS subnet group}"
AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-ecommerce-agent}"
AURORA_MIN_ACU="${AURORA_MIN_ACU:-0.5}"
AURORA_MAX_ACU="${AURORA_MAX_ACU:-1}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
CLUSTER_ID="${APP_NAME}-kb-aurora"
CLUSTER_ARN="arn:aws:rds:${AWS_REGION}:${ACCOUNT_ID}:cluster:${CLUSTER_ID}"
BUCKET="${APP_NAME}-kb-policies-${ACCOUNT_ID}"
ROLE_NAME="${APP_NAME}-kb-service-role"

echo "== 1/6: Aurora Serverless v2 cluster (${CLUSTER_ID}) =="
aws rds create-db-cluster \
  --db-cluster-identifier "$CLUSTER_ID" \
  --engine aurora-postgresql \
  --engine-version 16.14 \
  --engine-mode provisioned \
  --serverless-v2-scaling-configuration "MinCapacity=${AURORA_MIN_ACU},MaxCapacity=${AURORA_MAX_ACU}" \
  --master-username kbadmin \
  --manage-master-user-password \
  --db-subnet-group-name "$DB_SUBNET_GROUP_NAME" \
  --enable-http-endpoint \
  --database-name kbdb \
  --region "$AWS_REGION"

aws rds create-db-instance \
  --db-instance-identifier "${CLUSTER_ID}-instance-1" \
  --db-cluster-identifier "$CLUSTER_ID" \
  --engine aurora-postgresql \
  --db-instance-class db.serverless \
  --region "$AWS_REGION"

echo "Waiting for the instance to become available (usually 5-10 min)..."
aws rds wait db-instance-available --db-instance-identifier "${CLUSTER_ID}-instance-1" --region "$AWS_REGION"

SECRET_ARN="$(aws rds describe-db-clusters --db-cluster-identifier "$CLUSTER_ID" --region "$AWS_REGION" \
  --query "DBClusters[0].MasterUserSecret.SecretArn" --output text)"

echo "== 2/6: pgvector schema (via RDS Data API) =="
run_sql() {
  aws rds-data execute-statement --resource-arn "$CLUSTER_ARN" --secret-arn "$SECRET_ARN" \
    --database kbdb --sql "$1" --region "$AWS_REGION" >/dev/null
}
run_sql "CREATE EXTENSION IF NOT EXISTS vector;"
run_sql "CREATE SCHEMA IF NOT EXISTS bedrock_integration;"
run_sql "CREATE TABLE IF NOT EXISTS bedrock_integration.bedrock_kb (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), embedding vector(1024), chunks text, metadata json);"
run_sql "CREATE INDEX IF NOT EXISTS bedrock_kb_embedding_idx ON bedrock_integration.bedrock_kb USING hnsw (embedding vector_cosine_ops);"
# Bedrock KB requires the text column to have a full-text index -- discovered
# live, the first create-knowledge-base attempt below fails with a specific,
# actionable ValidationException naming this exact SQL if it's missing.
run_sql "CREATE INDEX IF NOT EXISTS bedrock_kb_chunks_idx ON bedrock_integration.bedrock_kb USING gin (to_tsvector('simple', chunks));"

echo "== 3/6: S3 bucket + policy docs =="
aws s3 mb "s3://${BUCKET}" --region "$AWS_REGION" || true
aws s3 cp "$(dirname "$0")/../policies" "s3://${BUCKET}/policies" --recursive

echo "== 4/6: IAM service role for the Knowledge Base =="
cat > /tmp/kb_trust_policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "bedrock.amazonaws.com" },
    "Action": "sts:AssumeRole",
    "Condition": {
      "StringEquals": { "aws:SourceAccount": "${ACCOUNT_ID}" },
      "ArnLike": { "aws:SourceArn": "arn:aws:bedrock:${AWS_REGION}:${ACCOUNT_ID}:knowledge-base/*" }
    }
  }]
}
EOF
aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file:///tmp/kb_trust_policy.json --region "$AWS_REGION" || true

cat > /tmp/kb_permissions_policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "S3ReadPolicies", "Effect": "Allow", "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${BUCKET}", "arn:aws:s3:::${BUCKET}/*"] },
    { "Sid": "BedrockInvokeEmbeddingModel", "Effect": "Allow", "Action": ["bedrock:InvokeModel"],
      "Resource": "arn:aws:bedrock:${AWS_REGION}::foundation-model/amazon.titan-embed-text-v2:0" },
    { "Sid": "RdsDataApi", "Effect": "Allow",
      "Action": ["rds-data:ExecuteStatement", "rds-data:BatchExecuteStatement", "rds:DescribeDBClusters"],
      "Resource": "${CLUSTER_ARN}" },
    { "Sid": "SecretsManagerAccess", "Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "${SECRET_ARN}" }
  ]
}
EOF
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name kb-access-policy \
  --policy-document file:///tmp/kb_permissions_policy.json --region "$AWS_REGION"

echo "Waiting for IAM propagation..."
sleep 15

echo "== 5/6: Knowledge Base + S3 data source =="
KB_ID="$(aws bedrock-agent create-knowledge-base \
  --name "${APP_NAME}-refund-policies" \
  --description "Bedrock Knowledge Base backing the support agent's policy retrieval." \
  --role-arn "arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}" \
  --knowledge-base-configuration "{\"type\":\"VECTOR\",\"vectorKnowledgeBaseConfiguration\":{\"embeddingModelArn\":\"arn:aws:bedrock:${AWS_REGION}::foundation-model/amazon.titan-embed-text-v2:0\"}}" \
  --storage-configuration "{\"type\":\"RDS\",\"rdsConfiguration\":{\"resourceArn\":\"${CLUSTER_ARN}\",\"credentialsSecretArn\":\"${SECRET_ARN}\",\"databaseName\":\"kbdb\",\"tableName\":\"bedrock_integration.bedrock_kb\",\"fieldMapping\":{\"primaryKeyField\":\"id\",\"vectorField\":\"embedding\",\"textField\":\"chunks\",\"metadataField\":\"metadata\"}}}" \
  --region "$AWS_REGION" --query "knowledgeBase.knowledgeBaseId" --output text)"

DS_ID="$(aws bedrock-agent create-data-source \
  --knowledge-base-id "$KB_ID" \
  --name policies-s3-source \
  --data-source-configuration "{\"type\":\"S3\",\"s3Configuration\":{\"bucketArn\":\"arn:aws:s3:::${BUCKET}\",\"inclusionPrefixes\":[\"policies/\"]}}" \
  --region "$AWS_REGION" --query "dataSource.dataSourceId" --output text)"

echo "== 6/6: Ingestion job =="
aws bedrock-agent start-ingestion-job --knowledge-base-id "$KB_ID" --data-source-id "$DS_ID" --region "$AWS_REGION"

echo ""
echo "Done. Set these on the agent (server env / agentcore deploy --env):"
echo "  BEDROCK_KNOWLEDGE_BASE_ID=${KB_ID}"
echo "  AURORA_CLUSTER_ARN=${CLUSTER_ARN}"
echo "  AURORA_SECRET_ARN=${SECRET_ARN}"
echo "  AURORA_DATABASE=kbdb"
