from langchain_aws import BedrockLLM
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import boto3

my_data = [
    "The Bruins play game 6 tonight."
    "Nikola Jokic is the NBA MVP."
    "The Celtics have a few days off to rest."
    "The Rangers won game 7 in the NHL against the Hurrincanes."
]

question = "Who can win the Stanley Cup?"

AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = BedrockLLM(model_id="amazon.titan-text-express-v1", client=bedrock)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

# create vector store
vector_store = FAISS.from_texts(my_data, bedrock_embeddings)

# create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}  # maybe we can add a score threshold here?
)

results = retriever.invoke(question)

results_string = []
for result in results:
    results_string.append(result.page_content)

# build template:
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the users question based on the following context: {context}",
        ),
        ("user", "{input}"),
    ]
)

chain = template.pipe(model)

response = chain.invoke({"input": question, "context": results_string})
print(response)
