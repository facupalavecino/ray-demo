module "s3_bucket_data" {
  source = "terraform-aws-modules/s3-bucket/aws"

  bucket = "${var.project}-data"

  attach_policy = true
  policy = jsonencode(
    {
      Version = "2012-10-17"
      Id      = "RayDemoS3BucketAccessPolicy"
      Statement = [
        {
          Sid    = "RayBucketAccessPolicy"
          Effect = "Allow"
          Principal = {
            AWS = aws_iam_role.ec2_role.arn
          }
          Action = [
            "s3:ListBucket"
          ]
          Resource = "arn:aws:s3:::${var.project}-data"
        },
        {
          Sid    = "RayBucketObjectsAccessPolicy"
          Effect = "Allow"
          Principal = {
            AWS = aws_iam_role.ec2_role.arn
          }
          Action = [
            "s3:GetObject",
            "s3:PutObject",
            "s3:DeleteObject"
          ]
          Resource = "arn:aws:s3:::${var.project}-data/*"
        }
      ]
    }
  )

  versioning = {
    enabled = false
  }

  tags = var.tags
}
