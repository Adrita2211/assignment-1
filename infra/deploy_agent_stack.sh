#!/usr/bin/env bash
#
# Deploys infra/cloudformation/agent-infra.yaml: ECR, ECS cluster/service
# (Fargate) with Service Auto Scaling, an ALB, RDS PostgreSQL + pgvector, and
# both IAM roles. Idempotent -- safe to re-run for updates without ever
# rolling the running image back to the bootstrap placeholder, since the
# currently-running image is always re-passed as a parameter override.
#
# Required env vars:
#   GITHUB_ORG        - your GitHub username or org
#   GITHUB_REPO       - the repo name this workflow lives in
#   GROQ_API_KEY      - a real Groq key, stored into SSM (not GitHub secrets)
#   DB_MASTER_PASSWORD - master password for the RDS instance
#
# Optional:
#   AWS_REGION        - defaults to us-east-1
#   APP_NAME          - defaults to ecommerce-support-agent
#   DB_INSTANCE_CLASS - defaults to db.t4g.micro (smallest that runs pgvector)
#   MIN_TASK_COUNT    - defaults to 1
#   MAX_TASK_COUNT    - defaults to 4
#   AUTOSCALING_CPU_TARGET - defaults to 60 (percent)

set -euo pipefail

: "${GITHUB_ORG:?set GITHUB_ORG}"
: "${GITHUB_REPO:?set GITHUB_REPO}"
: "${GROQ_API_KEY:?set GROQ_API_KEY}"
: "${DB_MASTER_PASSWORD:?set DB_MASTER_PASSWORD}"
AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-ecommerce-support-agent}"
DB_INSTANCE_CLASS="${DB_INSTANCE_CLASS:-db.t4g.micro}"
MIN_TASK_COUNT="${MIN_TASK_COUNT:-1}"
MAX_TASK_COUNT="${MAX_TASK_COUNT:-4}"
AUTOSCALING_CPU_TARGET="${AUTOSCALING_CPU_TARGET:-60}"
STACK_NAME="${APP_NAME}-infra"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Account: $ACCOUNT_ID   Region: $AWS_REGION   App: $APP_NAME"

OIDC_ARN=$(bash "$SCRIPT_DIR/setup_oidc_provider.sh" | tail -1)
echo "OIDC provider: $OIDC_ARN"

# ---------- SSM parameters (config/secrets -- deliberately kept outside the
# stack so rotating a key never triggers a stack update, and vice versa) ----------
aws ssm put-parameter --name "/${APP_NAME}/GROQ_API_KEY" --value "$GROQ_API_KEY" \
  --type SecureString --region "$AWS_REGION" --overwrite >/dev/null
echo "SSM parameter ready: /${APP_NAME}/GROQ_API_KEY"

for name in LANGFUSE_HOST LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
  aws ssm get-parameter --name "/${APP_NAME}/${name}" --region "$AWS_REGION" >/dev/null 2>&1 || \
    aws ssm put-parameter --name "/${APP_NAME}/${name}" --value "unset" --type SecureString \
      --region "$AWS_REGION" >/dev/null
done
echo "SSM placeholders ready for any LANGFUSE_* parameter not yet set (real values come from infra/deploy_langfuse_stack.sh)"

VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --region "$AWS_REGION" --query "Vpcs[0].VpcId" --output text)
SUBNET_IDS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC_ID" --region "$AWS_REGION" --query "Subnets[].SubnetId" --output text | tr '\t' ',')

# ---------- figure out which image to run: the currently-deployed one if
# the stack already exists, otherwise a bootstrap placeholder ----------
EXISTING_IMAGE=$(aws ecs describe-task-definition --task-definition "${APP_NAME}-task" --region "$AWS_REGION" \
  --query "taskDefinition.containerDefinitions[0].image" --output text 2>/dev/null || echo "None")
if [ "$EXISTING_IMAGE" != "None" ] && [ -n "$EXISTING_IMAGE" ]; then
  CONTAINER_IMAGE="$EXISTING_IMAGE"
  echo "Reusing currently-running image so this update can't roll the service back: $CONTAINER_IMAGE"
else
  CONTAINER_IMAGE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}:bootstrap"
  echo "No existing task definition found -- seeding with bootstrap placeholder (first real deploy comes from CI): $CONTAINER_IMAGE"
fi

aws cloudformation deploy \
  --template-file "$SCRIPT_DIR/cloudformation/agent-infra.yaml" \
  --stack-name "$STACK_NAME" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$AWS_REGION" \
  --parameter-overrides \
    AppName="$APP_NAME" \
    GitHubOrg="$GITHUB_ORG" \
    GitHubRepo="$GITHUB_REPO" \
    GitHubOidcProviderArn="$OIDC_ARN" \
    VpcId="$VPC_ID" \
    SubnetIds="$SUBNET_IDS" \
    ContainerImage="$CONTAINER_IMAGE" \
    DbInstanceClass="$DB_INSTANCE_CLASS" \
    DbMasterPassword="$DB_MASTER_PASSWORD" \
    MinTaskCount="$MIN_TASK_COUNT" \
    MaxTaskCount="$MAX_TASK_COUNT" \
    AutoscalingCpuTarget="$AUTOSCALING_CPU_TARGET"

DEPLOY_ROLE_ARN=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='GitHubDeployRoleArn'].OutputValue" --output text)
ALB_DNS=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='AlbDnsName'].OutputValue" --output text)
DB_ENDPOINT=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='DbEndpoint'].OutputValue" --output text)

echo ""
echo "==================================================================="
echo "Stack deployed: $STACK_NAME"
echo ""
echo "Stable URL (survives every redeploy):  http://${ALB_DNS}"
echo "RDS endpoint: ${DB_ENDPOINT}"
echo ""
echo "Next step -- seed pgvector once, from a machine that can reach the DB"
echo "(e.g. temporarily open DbSecurityGroup to your IP, or run from a task"
echo "inside the VPC):"
echo ""
echo "  export DATABASE_URL=postgresql://supportagent_admin:${DB_MASTER_PASSWORD}@${DB_ENDPOINT}:5432/supportagent"
echo "  python -m scripts.seed_pgvector"
echo ""
echo "Add these as GitHub repo variables (Settings > Secrets and variables"
echo "> Actions > Variables) -- the role ARN is safe as a plain variable,"
echo "OIDC trust is scoped to this exact repo:"
echo ""
echo "  AWS_DEPLOY_ROLE_ARN = ${DEPLOY_ROLE_ARN}"
echo "  AWS_REGION          = ${AWS_REGION}"
echo "==================================================================="
