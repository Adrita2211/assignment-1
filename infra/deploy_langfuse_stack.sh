#!/usr/bin/env bash
#
# CloudFormation-based replacement for the old setup_langfuse_ec2.sh.
# Deploys infra/cloudformation/langfuse-ec2.yaml (EC2 instance + Elastic IP +
# security group + IAM instance profile for SSM), then wires the resulting
# LangFuse credentials into this app's SSM parameters and forces the ECS
# agent to redeploy and pick them up.
#
# Optional env vars:
#   AWS_REGION     - defaults to us-east-1
#   APP_NAME       - defaults to ecommerce-support-agent
#   INSTANCE_TYPE  - defaults to t3.large

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-ecommerce-support-agent}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.large}"
STACK_NAME="${APP_NAME}-langfuse"

# See deploy_agent_stack.sh's comment on this same line -- pwd -W keeps
# --template-file in the Windows-path form aws.exe needs under Git Bash.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && { pwd -W 2>/dev/null || pwd; })"

echo "Region: $AWS_REGION   Instance type: $INSTANCE_TYPE"

VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --region "$AWS_REGION" --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC_ID" --region "$AWS_REGION" --query "Subnets[0].SubnetId" --output text)
MY_IP=$(curl -s https://checkip.amazonaws.com)

# Reuse existing secrets on a re-deploy (e.g. an instance-type bump)
# instead of generating fresh ones every time, which would orphan the old
# LangFuse project's data from the app's perspective.
EXISTING=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" 2>/dev/null || echo "")
if [ -n "$EXISTING" ]; then
  echo "Stack already exists — this script only supports fresh creation today;"
  echo "for a plain infra update (e.g. instance type), edit and re-run"
  echo "'aws cloudformation deploy' directly with the same parameters."
  echo "To fully recreate, run infra/teardown_langfuse_stack.sh first."
  exit 1
fi

SALT=$(openssl rand -hex 16)
ENCRYPTION_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 16)
CLICKHOUSE_PASSWORD=$(openssl rand -hex 16)
MINIO_ROOT_PASSWORD=$(openssl rand -hex 16)
REDIS_AUTH=$(openssl rand -hex 16)
NEXTAUTH_SECRET=$(openssl rand -hex 16)
INIT_PROJECT_PUBLIC_KEY="pk-lf-$(openssl rand -hex 12)"
INIT_PROJECT_SECRET_KEY="sk-lf-$(openssl rand -hex 12)"
INIT_USER_PASSWORD=$(openssl rand -hex 8)

aws cloudformation deploy \
  --template-file "$SCRIPT_DIR/cloudformation/langfuse-ec2.yaml" \
  --stack-name "$STACK_NAME" \
  --capabilities CAPABILITY_IAM \
  --region "$AWS_REGION" \
  --parameter-overrides \
    AppName="$APP_NAME" \
    InstanceType="$INSTANCE_TYPE" \
    VpcId="$VPC_ID" \
    SubnetId="$SUBNET_ID" \
    MyIpCidr="${MY_IP}/32" \
    Salt="$SALT" \
    EncryptionKey="$ENCRYPTION_KEY" \
    PostgresPassword="$POSTGRES_PASSWORD" \
    ClickhousePassword="$CLICKHOUSE_PASSWORD" \
    MinioRootPassword="$MINIO_ROOT_PASSWORD" \
    RedisAuth="$REDIS_AUTH" \
    NextAuthSecret="$NEXTAUTH_SECRET" \
    InitProjectPublicKey="$INIT_PROJECT_PUBLIC_KEY" \
    InitProjectSecretKey="$INIT_PROJECT_SECRET_KEY" \
    InitUserPassword="$INIT_USER_PASSWORD"

LANGFUSE_URL=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='LangfuseUrl'].OutputValue" --output text)

echo ""
echo "Stack deployed. Waiting for the instance to finish installing Docker"
echo "and booting all six containers (usually 3-5 minutes)..."
ATTEMPTS=0
until curl -s --max-time 5 "${LANGFUSE_URL}/api/public/health" 2>/dev/null | grep -q '"status":"OK"'; do
  ATTEMPTS=$((ATTEMPTS + 1))
  if [ "$ATTEMPTS" -gt 60 ]; then
    echo "Still not healthy after 10 minutes — check /var/log/user-data.log on the" >&2
    echo "instance via SSM Session Manager before continuing." >&2
    exit 1
  fi
  sleep 10
done
echo "Healthy: ${LANGFUSE_URL}"

# ---------- wire the real credentials into the agent's SSM parameters and
# force a redeploy so the running ECS agent picks them up ----------
aws ssm put-parameter --name "/${APP_NAME}/LANGFUSE_HOST" --value "$LANGFUSE_URL" \
  --type SecureString --region "$AWS_REGION" --overwrite >/dev/null
aws ssm put-parameter --name "/${APP_NAME}/LANGFUSE_PUBLIC_KEY" --value "$INIT_PROJECT_PUBLIC_KEY" \
  --type SecureString --region "$AWS_REGION" --overwrite >/dev/null
aws ssm put-parameter --name "/${APP_NAME}/LANGFUSE_SECRET_KEY" --value "$INIT_PROJECT_SECRET_KEY" \
  --type SecureString --region "$AWS_REGION" --overwrite >/dev/null
echo "Updated /${APP_NAME}/LANGFUSE_* SSM parameters"

if aws ecs describe-services --cluster "${APP_NAME}-cluster" --services "${APP_NAME}-service" \
    --region "$AWS_REGION" --query "services[?status=='ACTIVE']" --output text 2>/dev/null | grep -q .; then
  aws ecs update-service --cluster "${APP_NAME}-cluster" --service "${APP_NAME}-service" \
    --force-new-deployment --region "$AWS_REGION" >/dev/null
  echo "Forced ECS redeploy to pick up the new LangFuse credentials"
fi

echo ""
echo "==================================================================="
echo "LangFuse:  ${LANGFUSE_URL}"
echo "Dashboard login: admin@ecommerce-support-agent.local / ${INIT_USER_PASSWORD}"
echo ""
echo "Also add these as GitHub repo secrets if online_eval.py runs from CI:"
echo "  LANGFUSE_HOST=${LANGFUSE_URL}"
echo "  LANGFUSE_PUBLIC_KEY=${INIT_PROJECT_PUBLIC_KEY}"
echo "  LANGFUSE_SECRET_KEY=${INIT_PROJECT_SECRET_KEY}"
echo "==================================================================="
