data "terraform_remote_state" "networking" {
  backend = "s3"

  config = {
    bucket = "facu-terraform-statefiles"
    key    = "ray-demo/networking/terraform.tfstate"
    region = "us-east-1"
    profile = "facu-fullaccess"
  }
}
