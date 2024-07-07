variable "tags" {
  type        = map(any)
  description = "Default tags"
  default     = {
    Project = "ray-demo"
  }
}
