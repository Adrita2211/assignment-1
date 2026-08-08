#!/usr/bin/env bash
#
# Tears down the LangFuse CloudFormation stack — instance, Elastic IP,
# security group, and IAM instance profile all go in one command. No more
# manually remembering to release the EIP (an unassociated one is billed
# hourly) — CloudFormation deletes everything it created, in the right
# order, every time.

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-ecommerce-support-agent}"
STACK_NAME="${APP_NAME}-langfuse"

echo "Deleting stack: $STACK_NAME"
aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$AWS_REGION"
aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"
echo "Stack deleted."
