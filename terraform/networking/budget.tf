resource "aws_budgets_budget" "monthly_budget" {
  name              = "Ray Demo Budget"
  budget_type       = "COST"
  limit_amount      = "25"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"
  time_period_start = "2024-07-07_00:00"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = ["palav.facundo@gmail.com"]
  }

  cost_filter {
    name = "TagKeyValue"
    values = [
      "Project$ray-demo",
    ]
  }

  tags = var.tags
}
