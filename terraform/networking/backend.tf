terraform {
  required_version = ">= 0.12.0"

  backend "s3" {
    bucket = "facu-terraform-statefiles"
    key    = "ray-demo/networking/terraform.tfstate"
    region = "us-east-1"
    profile = "facu-fullaccess"
  }
}
