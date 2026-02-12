# https://community.aws/content/2oa6QAE7tgxmAsgAExXY2B2MqCV/getting-started-with-bedrock-application-inference-profile
# Script to discover/create inference profile (account, model-id) and tag to model invocations
import boto3
import json

'''
AWS Account: tc-mcle-001
Policy: bedrock-finops-sk255251
Role: bedrock-finops-sk255251
CloudWatch LogGroup: /aws/bedrock/bedrock-finops-sk255251 (Required to route logs, metrics)
'''

# Test Simulation Constants
region = "us-west-2"
chatModel = "anthropic.claude-3-5-sonnet-20241022-v2:0"
embedModel = "amazon.titan-embed-text-v2:0"
accountTagName = "teracloud:account"
modelTagName = "teracloud:component:type"
Questions = [
    "Should I be storing documents in Amazon S3 or EFS for cost effective applications?",
    "What is the future affect of inflation on interest rates?",
    "Should I use S3 Intelligent-Tiering or Glacier for long-term backups?",
    "What is the most cost-effective way to store and retrieve infrequently accessed files in AWS?"
]

# Preparing Inference Profile Tag Values
accountTagValue = "acc-41"
chatModelTagValue = f"{accountTagValue}-claude-3.5"
embedModelTagValue = f"{accountTagValue}-embed-2.0"

# Bedrock Initialization
session = boto3.Session()
bedrock = session.client("bedrock", region_name=region)
bedrock_runtime = session.client("bedrock-runtime", region_name=region)

# Extract MODEL ARN
def get_model_arn(model_id: str) -> str:
    models = bedrock.list_foundation_models()
    for model in models["modelSummaries"]:
        if model["modelId"] == model_id:
            return model["modelArn"]
    raise ValueError(f"Model ID '{model_id}' not found.")
chatModelARN = get_model_arn(chatModel)
embedModelARN = get_model_arn(embedModel)
print(f"Chat Model ARN: {chatModelARN}")
print(f"Embed Model ARN: {embedModelARN}")

# Lists USER Defined Application Inference Profiles
usr_def_pf = bedrock.list_inference_profiles(typeEquals='APPLICATION').get("inferenceProfileSummaries", [])

# Discovery/Creation of application inference profile
def get_or_create_inference_profile(ModelTagValue, ModelARN):
    # Discover Profile
    for profile in usr_def_pf:
        tags = bedrock.list_tags_for_resource(resourceARN=profile["inferenceProfileArn"]).get("tags", [])
        tag_dict = {tag["key"]: tag["value"] for tag in tags}
        if tag_dict.get(accountTagName) == accountTagValue and tag_dict.get(modelTagName) == ModelTagValue:
            return profile["inferenceProfileArn"]

    # Create Profile
    inf_profile_response = bedrock.create_inference_profile(
        inferenceProfileName=f"InferenceProfile{accountTagValue}",
        modelSource={
            'copyFrom': ModelARN
        },
        tags=[
            {"key": accountTagName, "value": accountTagValue},
            {"key": modelTagName, "value": ModelTagValue},
        ]
    )
    return inf_profile_response.get("inferenceProfileArn")
chatModelInferenceProfileARN = get_or_create_inference_profile(chatModelTagValue, chatModelARN)
embedModelInferenceProfileARN = get_or_create_inference_profile(embedModelTagValue, embedModelARN)
print("Chat Model Inference Profile ARN: ", chatModelInferenceProfileARN)
print("Embed Model Inference Profile ARN: ", embedModelInferenceProfileARN)

# Invoke Model API
chat_context = "You are an expert on AWS services and always try to provide correct and concise answers."
# Invoking Chat Model
print("\n" + "=" * 100)
print("Invoking Chat Model")
print("=" * 100 + "\n")
for Question in Questions:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
        "system": chat_context,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{Question}",
                    }
                ]
            }
        ]
    })
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=chatModelInferenceProfileARN,
        accept='application/json', contentType='application/json'
    )
    model_response = json.loads(response.get('body').read())
    print("\n" + "=" * 80)
    print(f"Question Asked:\n {Question}\n")
    print("Model Response:")
    print(model_response.get('content')[0].get('text'))
    print("=" * 80 + "\n")

# Invoking Embed Model
print("\n" + "=" * 100)
print("Invoking Embed Model")
print("=" * 100 + "\n")
for i in range(1):
    body = json.dumps({
        "inputText": "Artificial Intelligence (AI) has rapidly emerged as one of the most transformative technologies of the 21st century. From powering voice assistants to revolutionizing healthcare, AI is reshaping industries, societies, and human potential. At its core, AI refers to the simulation of human intelligence in machines that are capable of learning, reasoning, problem-solving, and even creativity. As the technology continues to evolve, its impact on the world grows deeper and more complex."
                     "AI's development can be traced back to the mid-20th century, but it is the recent advances in computational power, data availability, and algorithmic design that have catapulted it into mainstream use. Machine learning, a subset of AI, enables systems to learn from data and improve over time without being explicitly programmed. Deep learning, which uses neural networks to process large volumes of data, has enabled breakthroughs in image recognition, natural language processing, and autonomous systems."
                     "One of the most prominent applications of AI is in the healthcare industry. AI-driven diagnostic tools can analyze medical images with remarkable accuracy, sometimes even outperforming human doctors. Predictive analytics can help identify patients at risk of certain diseases, enabling earlier interventions. In drug discovery, AI accelerates the identification of potential compounds, drastically reducing development time and cost."
                     "In business, AI enhances decision-making and efficiency. Companies use AI to forecast demand, optimize supply chains, and personalize customer experiences. Chatbots and virtual assistants handle customer service inquiries around the clock, while fraud detection algorithms monitor transactions in real time. AI is not just a tool for automation; it is a catalyst for innovation."
                     "However, the rise of AI is not without challenges. Ethical concerns are at the forefront of the AI debate. Issues such as bias in algorithms, lack of transparency in decision-making (often called the 'black box' problem), and the potential for mass surveillance raise questions about accountability and fairness. Furthermore, the impact of AI on employment is a subject of ongoing concern. While AI creates new job opportunities in tech and data science, it also threatens to displace workers in routine and repetitive roles."
                     "Governments, institutions, and technology leaders must work together to ensure AI is developed and deployed responsibly. Establishing clear regulations, promoting transparency, and investing in AI literacy and workforce retraining are critical steps toward harnessing AI for good. Global cooperation will also be essential to address cross-border challenges and ensure that the benefits of AI are equitably distributed."
                     "In conclusion, Artificial Intelligence is not just a technological advancement—it is a paradigm shift. It holds immense potential to solve some of humanity’s greatest challenges, from climate change to global health. However, its power must be balanced with responsibility. As we stand at the threshold of the AI age, the decisions we make today will shape the world of tomorrow.",
        "dimensions": 1024,
        "normalize": True
    })
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=embedModelInferenceProfileARN,
        accept='application/json', contentType='application/json'
    )
    model_response = json.loads(response.get('body').read())
    print("\n" + "=" * 80)
    print("Model Response:")
    print("Full Response:", json.dumps(model_response, indent=2))
    print("=" * 80 + "\n")

# Delete Inference Profile
# try:
#     delete_res = bedrock.delete_inference_profile(inferenceProfileIdentifier=inf_profile_arn)
#     if delete_res.get("ResponseMetadata", {}).get("HTTPStatusCode", 500) == 200:
#         print("Inference Profile deleted successfully.")
# except ClientError as e:
#     print(f"AWS ClientError: {e.response['Error']['Message']}")
# except Exception as e:
#     print(f"Unexpected error: {str(e)}")