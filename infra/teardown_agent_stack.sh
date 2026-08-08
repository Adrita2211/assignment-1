#!/usr/bin/env bash
#
# Tears down the agent-infra CloudFormation stack -- ECS service/cluster,
# ALB/target-group/listener, security groups, RDS instance, and both IAM
# roles, all in one command, in dependency order. Also removes the SSM
# parameters, which live outside the stack on purpose (see
# deploy_agent_stack.sh). The GitHub OIDC provider is never touched (see
# setup_oidc_provider.sh).
#
# Empty the ECR repo first -- CloudFormation won't delete a non-empty
# repository, a real gotcha called out in the assignment's own §7.5.

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-ecommerce-support-agent}"
STACK_NAME="${APP_NAME}-infra"

echo "Emptying ECR repository ${APP_NAME} (if it exists)..."
IMAGE_TAGS=$(aws ecr list-images --repository-name "$APP_NAME" --region "$AWS_REGION" \
  --query 'imageIds[*]' --output json 2>/dev/null || echo "[]")
if [ "$IMAGE_TAGS" != "[]" ]; then
  aws ecr batch-delete-image --repository-name "$APP_NAME" --region "$AWS_REGION" \
    --image-ids "$IMAGE_TAGS" >/dev/null
  echo "Deleted all images in ${APP_NAME}"
fi

echo "Deleting stack: $STACK_NAME"
aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$AWS_REGION"
aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"
echo "Stack deleted."

for name in GROQ_API_KEY LANGFUSE_HOST LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
  aws ssm delete-parameter --name "/${APP_NAME}/${name}" --region "$AWS_REGION" >/dev/null 2>&1 \
    && echo "Deleted SSM parameter: ${name}" || echo "No SSM parameter ${name} to delete"
done

echo ""
echo "Teardown complete. (GitHub OIDC provider left in place, intentionally.)"
