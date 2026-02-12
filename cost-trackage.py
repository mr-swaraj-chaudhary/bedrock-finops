# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ce/client/get_cost_and_usage.html
# https://aws.amazon.com/bedrock/pricing/
# Script to track model invocation costs (Mocking Tenant Design)
"""
		             (Tags)
T1 - T1.C - T1.C.I - acc-41, acc-41-claude-3.5
   - T1.E - T1.E.I - acc-41, acc-41-embed-2.0
"""

import boto3
import datetime

# Define timeline
endDate = datetime.date.today()
startDate = endDate - datetime.timedelta(days=30)

# Tag Keys
accountTagName = "teracloud:account"
modelTagName = "teracloud:component:type"

# Tag Values
accountTagValue = "acc-41"
chatModelTagValue = f"{accountTagValue}-claude-3.5"
embedModelTagValue = f"{accountTagValue}-embed-2.0"

# Fetch costs for inference profiles with the specified tag
ce_client = boto3.client("ce")
def get_inference_profile_cost(tagKey, tagValue):
    """
    Fetch cost details for AWS Bedrock Inference Profiles with specific tag
    """
    response = ce_client.get_cost_and_usage(
        TimePeriod={
            "Start": startDate.strftime("%Y-%m-%d"),
            "End": endDate.strftime("%Y-%m-%d"),
        },
        Granularity="DAILY",  # Options: DAILY, MONTHLY
        Metrics=["UnblendedCost"],  # Cost before discounts & credits
        GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
        Filter={
            "And": [
                {
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": [
                            "Amazon Bedrock",
                            "Claude (Amazon Bedrock Edition)",
                            "Claude 3 Haiku (Amazon Bedrock Edition)",
                            "Claude 3.5 Sonnet v2 (Amazon Bedrock Edition)",
                            "Claude Instant (Amazon Bedrock Edition)"
                        ]
                    }
                },
                {
                    "Tags": {
                        "Key": tagKey,
                        "Values": [tagValue]
                    }
                }
            ]
        }
    )
    return response["ResultsByTime"]

# Print cost summary
def print_cost_summary(costData, tagValue):
    print(f"\n💰 Cost Summary for Inference Profile tagged with **{tagValue}**:\n")
    for day in costData:
        date = day["TimePeriod"]["Start"]
        totalCost = float(day["Total"].get("UnblendedCost", {}).get("Amount", 0))

        inputCost, outputCost = 0, 0
        for group in day.get("Groups", []):
            key = group["Keys"][0]
            keyCost = float(group["Metrics"]["UnblendedCost"]["Amount"])
            if "InputTokenCount" in key or "input-tokens" in key:
                inputCost += keyCost
            elif "OutputTokenCount" in key or "output-tokens" in key:
                outputCost += keyCost

        # Ensure total_cost includes grouped costs in case 'Total' is missing
        if not totalCost:
            totalCost = inputCost + outputCost

        print(f"📅 {date} → 🟢 Input Token Usage Cost: ${inputCost}, 🔵 Output Token Usage Cost: ${outputCost}, 💰 Total Costs: ${totalCost}")

chatModelCosts = get_inference_profile_cost(modelTagName, chatModelTagValue)
embedModelCosts = get_inference_profile_cost(modelTagName, embedModelTagValue)
accountCosts = get_inference_profile_cost(accountTagName, accountTagValue)

print_cost_summary(chatModelCosts, chatModelTagValue)
print_cost_summary(embedModelCosts, embedModelTagValue)
print_cost_summary(accountCosts, accountTagValue)