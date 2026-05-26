import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# 1. Unlock the Vaults
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("coliving-rules") # Make sure this matches the name you used!

# 2. Read the Raw Text
print("Reading house rules...")
with open("house_rules.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# 3. Chop the text into paragraphs (Chunks)
# We split the text every time there is a double line break
chunks = raw_text.split("\n\n")
chunks = [c.strip() for c in chunks if c.strip()] # Clean up empty spaces
print(f"Chopped into {len(chunks)} paragraphs.")

# 4. Translate to Math (Embeddings) and Package it
vectors_to_upload = []
print("Translating text into vector embeddings... This takes a few seconds.")

for i, chunk in enumerate(chunks):
    # Call OpenAI to turn the text into 1,536 numbers
    response = openai_client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small" 
    )
    embedding = response.data[0].embedding
    
    # Create the package: [ID, Math Coordinates, Original Text]
    vectors_to_upload.append({
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {"text": chunk} # We hide the original text in the metadata!
    })

# 5. Send it to Pinecone
print("Uploading to Pinecone...")
index.upsert(vectors=vectors_to_upload)

print("✅ Success! Your database is now populated and ready.")
