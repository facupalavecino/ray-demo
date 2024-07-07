output "vpc_id" {
    description = "The ID of the VPC"
    value = aws_vpc.main.id
}

output "subnet_id" {
    description = "The ID of the public subnet"
    value = aws_subnet.public.id
}

output "private_subnet_ids" {
    description = "The IDs of the private subnets"
    value = [aws_subnet.private1.id, aws_subnet.private2.id]
}

output "public_subnet_ids" {
    description = "The IDs of the public subnets"
    value = [aws_subnet.public.id, aws_subnet.public2.id]
}
