import pymongo
import os
import requests
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
client = pymongo.MongoClient(mongo_uri)
db = client.sample_mflix
collection = db.movies


hf_token = os.getenv("HUGGING_FACE_API")
embedding_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def generate_embedding(text: str) -> list[float]:
    # This function should call the Hugging Face API to generate an embedding for the given text.
    # For demonstration purposes, we'll return a dummy embedding.
    
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text}
    )

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    
    return response.json()


for doc in collection.find({"plot": {"$exists":True}}).limit(40):
    plot = doc["plot"]
    doc['plot_embedding_hf'] = generate_embedding(plot)
    collection.replace_one({"_id": doc["_id"]}, doc)

    print(f"Movie: {doc['title']}, Embedding: {doc['plot_embedding_hf'][:5]}...")  # Print the first 5 values of the embedding for brevity

query = "imaginary characters from outer space at war"

results  = collection.aggregate(
    [
        {"$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 5,
            "index": "vector_index"}
        }
    ]
)

for doc in results:
    print(f"Movie name: {doc['title']},\nMovie Plot: {doc['plot']}")  # Print the first 10 characters of the plot for brevity