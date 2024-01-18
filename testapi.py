from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load Sentence Transformers model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to calculate similarity
def get_similarity(t1, t2):
    # Compute embedding for both texts
    embedding_1 = model.encode(t1, convert_to_tensor=True)
    embedding_2 = model.encode(t2, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(embedding_1, embedding_2).item()

    # Normalize similarity between 0 and 1
    normalized_similarity = 0.5 * (similarity_score + 1)

    return normalized_similarity

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
