from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

origins = ["*"]
app.add_middleware(
 CORSMiddleware,
 allow_origins=origins,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

# Load Sentence Transformers model
model = SentenceTransformer('stsb-roberta-base')

# Function to calculate similarity
def get_similarity(t1, t2):
    # Convert text to embeddings
    embedding_1 = model.encode(t1, convert_to_tensor=True)
    embedding_2 = model.encode(t2, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(embedding_1, embedding_2).item()

    return similarity_score

# API endpoint to calculate similarity
@app.post('/calculate_similarity')
def calculate_similarity(data: dict):
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    if text1 and text2:
        similarity_score = get_similarity(text1, text2)
        return {'similarity_score': similarity_score}
    else:
        raise HTTPException(status_code=400, detail='Please provide both text1 and text2')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
