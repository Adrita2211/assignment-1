#!/usr/bin/env bash
#
# Prints the current public IP of the running ECS Fargate task. There's no
# load balancer in front of this demo on purpose (keeps AWS cost/complexity
# down for a teardown-after-use setup) — but that means the IP changes on
# every new deployment. Run this right before each online_eval.py round.
#
# Usage: infra/get_task_ip.sh
#   ONLINE_EVAL_URL=$(infra/get_task_ip.sh)
#   python -m eval.online_eval --url "http://${ONLINE_EVAL_URL}:8080" --rounds 3

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-ecommerce-support-agent}"

TASK_ARN=$(aws ecs list-tasks --cluster "${APP_NAME}-cluster" --service-name "${APP_NAME}-service" \
  --region "$AWS_REGION" --query "taskArns[0]" --output text)

if [ "$TASK_ARN" = "None" ] || [ -z "$TASK_ARN" ]; then
  echo "No running task found in ${APP_NAME}-cluster / ${APP_NAME}-service" >&2
  exit 1
fi

ENI_ID=$(aws ecs describe-tasks --cluster "${APP_NAME}-cluster" --tasks "$TASK_ARN" --region "$AWS_REGION" \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text)

PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI_ID" --region "$AWS_REGION" \
  --query "NetworkInterfaces[0].Association.PublicIp" --output text)

echo "$PUBLIC_IP"
