#!/bin/bash
set -e

REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --profile personal --query Account --output text)
REPO=housing
IMAGE_URI=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Authenticating with ECR..."
aws ecr get-login-password --profile personal --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

echo "Building image..."
docker build --platform linux/amd64 -t $REPO $PROJECT_DIR

echo "Tagging image..."
docker tag $REPO:latest $IMAGE_URI

echo "Pushing to ECR..."
docker push $IMAGE_URI

echo "Done: $IMAGE_URI"
