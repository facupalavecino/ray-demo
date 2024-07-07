variable "tags" {
  type        = map(any)
  description = "Default tags"
  default     = {
    Project = "ray-demo"
  }
}

variable "project" {
  type = string
  default = "ray-demo"
  description = "Name of the project"
}
