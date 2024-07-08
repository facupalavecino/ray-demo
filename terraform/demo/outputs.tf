output "ray_cluster_security_group_id" {
  value = aws_security_group.ray_cluster_sg.id
  description = "The ID of the security group for the Ray cluster"
}
