data "aws_iam_policy" "ec2_fullaccess" {
  name = "AmazonEC2FullAccess"
}

data "aws_iam_policy" "s3_fullaccess" {
  name = "AmazonS3FullAccess"
}

resource "aws_iam_role" "ec2_role" {
  name        = "${var.project}-ec2-role"
  description = "Role assumed by the EC2 instances"

  assume_role_policy = jsonencode(
    {
      Version = "2012-10-17"
      Statement = [
        {
          Action = "sts:AssumeRole"
          Effect = "Allow"
          Sid    = ""
          Principal = {
            Service = "ec2.amazonaws.com"
          }
        }
      ]
    }
  )

  tags = var.tags
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project}-ec2-profile"

  role = aws_iam_role.ec2_role.name

  tags = var.tags
}

resource "aws_iam_policy" "iam_access" {
  name = "${var.project}-ec2-iam-access"
  
  description = "Allows the ec2 instance to pass the role to another instances"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = "arn:aws:iam::*:role/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "iam_access" {
  role = aws_iam_role.ec2_role.id
  policy_arn = aws_iam_policy.iam_access.arn
}

resource "aws_iam_role_policy_attachment" "ec2_fullaccess" {
  role = aws_iam_role.ec2_role.id
  policy_arn = data.aws_iam_policy.ec2_fullaccess.arn
}

resource "aws_iam_role_policy_attachment" "s3_fullaccess" {
  role = aws_iam_role.ec2_role.id
  policy_arn = data.aws_iam_policy.s3_fullaccess.arn
}
