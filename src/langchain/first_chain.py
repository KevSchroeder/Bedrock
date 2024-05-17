from langchain_aws import BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
import boto3

AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = BedrockLLM(model_id="amazon.titan-text-express-v1", client=bedrock)


def invoke_model():
    response = model.invoke("What is a sports arbitrage?")
    print(response)


def fist_chain():
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write a short description for the product provided by the user",
            ),
            ("human", "{product_name}"),
        ]
    )
    chain = template.pipe(model)

    response = chain.invoke({"product_name": "sports arbitrage"})
    print(response)


fist_chain()