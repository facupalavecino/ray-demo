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
