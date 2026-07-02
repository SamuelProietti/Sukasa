import os
import json
import requests
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# 1. Load Environment Variables (This works perfectly here)
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("coliving-rules")

app = Flask(__name__)

# 2. Load the System Prompt safely
def load_personality():
    with open("system_prompt.txt", "r", encoding="utf-8") as file:
        return file.read()

# 3. The Google Maps Function (With the official Link fix)
def search_google_maps(query, location="Poblenou, Barcelona, Spain"):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    url = f"https://places.googleapis.com/v1/places:searchText?key={api_key}"
    search_text = f"{query} near {location}"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.googleMapsUri"
    }
    payload = {"textQuery": search_text}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        
        if "places" in data:
            results = []
            for place in data["places"][:3]: 
                name = place.get("displayName", {}).get("text", "Unknown")
                address = place.get("formattedAddress", "No address")
                maps_url = place.get("googleMapsUri", "No link available")
                results.append(f"- **{name}**: {address}\n📍 Map: {maps_url}")
            return "\n\n".join(results)
        else:
            return "No places found right now."
    except Exception as e:
        return f"Maps error: {e}"

# 4. The Tool Definition
agent_tools = [{
    "type": "function",
    "function": {
        "name": "search_google_maps",
        "description": "Searches live Google Maps data for locations, stores, or restaurants.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The place to search for"}
            },
            "required": ["query"]
        }
    }
}]

# 5. The Twilio Webhook (Where WhatsApp connects)
@app.route('/whatsapp', methods=['POST'])
def whatsapp_reply():
    incoming_msg = request.values.get('Body', '').strip()
    num_media = int(request.values.get('NumMedia', 0))
    
    resp = MessagingResponse()
    msg = resp.message()

    # GUARDRAIL: Block Voice Notes and Images so they don't crash the bot
    if num_media > 0 or incoming_msg == "":
        msg.body("Oops! 🙈 Right now I can only read text messages. Please type out your question!")
        return str(resp)

    # --- RAG PIPELINE ---
    try:
        raw_query_embedding = openai_client.embeddings.create(
            input=incoming_msg,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        search_results = index.query(
            vector=raw_query_embedding,
            top_k=3,
            include_metadata=True
        )
        retrieved_context = "\n\n".join([match["metadata"]["text"] for match in search_results["matches"]])
    except:
        retrieved_context = "No specific house rules found."

    # --- OPENAI LOGIC ---
    messages_for_openai = [
        {"role": "system", "content": load_personality() + f"\n\nHouse Rules Context:\n{retrieved_context}"},
        {"role": "user", "content": incoming_msg}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages_for_openai,
        tools=agent_tools,
        tool_choice="auto"
    )
    
    ai_message = response.choices[0].message

    if ai_message.tool_calls:
        messages_for_openai.append(ai_message)
        
        for tool_call in ai_message.tool_calls:
            if tool_call.function.name == "search_google_maps":
                arguments = json.loads(tool_call.function.arguments)
                maps_results = search_google_maps(arguments.get("query"))
                
                messages_for_openai.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"{maps_results}\n\nCRITICAL INSTRUCTION: You MUST print the exact raw URL provided in the data."
                })
        
        final_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages_for_openai
        )
        msg.body(final_response.choices[0].message.content)
    else:
        msg.body(ai_message.content)

    return str(resp)

if __name__ == '__main__':
    app.run(port=5000)