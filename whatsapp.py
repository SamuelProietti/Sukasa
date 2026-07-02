import os
import json
import requests
from dotenv import load_dotenv
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
from pinecone import Pinecone

# --- 1. SETUP & SECRETS ---
load_dotenv()
app = Flask(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("coliving-rules")

# 🧠 NEW: This dictionary will hold the chat history for every phone number
# Structure: { "+34600000000": [messages], "+56900000000": [messages] }
conversation_memory = {}

# --- 2. YOUR GOOGLE MAPS TOOL ---
def search_google_maps(query, location="Poblenou, Barcelona, Spain"):
    """Pings the Google Places API to find live local recommendations."""
    api_key = os.getenv("GOOGLE_API_KEY")
    url = f"https://places.googleapis.com/v1/places:searchText?key={api_key}"
    search_text = f"{query} near {location}"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.googleMapsUri"
    }
    payload = {"textQuery": search_text}
    print(f"🚨 SENDING REQUEST TO: {url}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        
        if "places" in data:
            results = []
            # 1. Grab only the absolute best match
            for place in data["places"][:1]: 
                name = place.get("displayName", {}).get("text", "Unknown")
                map_link = place.get("googleMapsUri", "No link available")
                
                # 2. Keep it super clean: Just the name and the link
                results.append(f"📍 **{name}**\n{map_link}")
            return "\n".join(results)
            
        elif "error" in data:
            return f"**Google API Error:** {data['error'].get('message', 'Unknown error')}"
            
        else:
            return f"Google returned no places. Raw data: {data}"
            
    except Exception as e:
        return f"Error connecting to Maps: {e}"

agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_google_maps",
            "description": "Searches live Google Maps data. Use this whenever the user asks for the nearest location of ANYTHING, including stores, restaurants, beaches, parks, or metro stations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The type of place to search for (e.g., 'Zara', 'coffee shop', 'pharmacy')"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# --- 3. LOAD YOUR PERSONALITY ---
def load_personality():
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as file:
            return file.read()
    except Exception:
        return "You are Sukasa, a helpful Smart Host in Barcelona."

# Render Health Check Front Door
@app.route('/', methods=['GET'])
def health_check():
    return "Sukasa Bot is alive and running!", 200
    
# --- 4. THE WHATSAPP WEBHOOK ---
@app.route('/whatsapp', methods=['POST'])
def whatsapp_reply():
    # 📞 NEW: Grab the unique WhatsApp phone number of the person texting us
    guest_phone = request.values.get('From', '')
    incoming_msg = request.values.get('Body', '')
    print(f"📩 [{guest_phone}] asked: {incoming_msg}")
    
    try:
        # A. RAG PIPELINE (Check house rules!)
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
        
        # B. MEMORY MANAGMENT FOR THIS SPECIFIC PHONE NUMBER
        # If this phone number is texting us for the first time, initialize their history
        if guest_phone not in conversation_memory:
            conversation_memory[guest_phone] = []
        
        # Append the new message to this guest's specific running history
        conversation_memory[guest_phone].append({"role": "user", "content": incoming_msg})
        
        # Keep memory clean: Only remember the last 10 messages so the payload doesn't get huge
        if len(conversation_memory[guest_phone]) > 10:
            conversation_memory[guest_phone] = conversation_memory[guest_phone][-10:]
            
        # C. BUILD THE FULL CONTEXT FOR OPENAI
        # --- C. BUILD THE FULL CONTEXT FOR OPENAI ---
        system_rules = (
            f"{load_personality()}\n\n"
            f"Here is the reference property context from the database:\n{retrieved_context}\n\n"
            f"CRITICAL RULE: If any information in the database context conflicts with your core personality "
            f"instructions (such as lockout fees, lockbox codes, or handling your empadronamiento), "
            f"the core personality instructions MUST take absolute priority. Do not use old rules from the database.\n\n"
            f"If the user asks for local stores, restaurants, or businesses, use your Google Maps tool to find them."
        )
        
        # Combine the system instructions with this specific guest's entire chat history
        messages_for_openai = [{"role": "system", "content": system_rules}] + conversation_memory[guest_phone]
        
        # D. THE AGENTIC LOOP
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_openai,
            tools=agent_tools,
            tool_choice="auto" 
        )
        
        ai_message = response.choices[0].message
        
        # E. DID THE AI USE GOOGLE MAPS?
        if ai_message.tool_calls:
            messages_for_openai.append(ai_message)
            
            for tool_call in ai_message.tool_calls:
                if tool_call.function.name == "search_google_maps":
                    arguments = json.loads(tool_call.function.arguments)
                    search_query = arguments.get("query")
                    
                    print(f"📍 Checking live map for: {search_query}...")
                    maps_results = search_google_maps(search_query)
                    
                    messages_for_openai.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": maps_results
                    })
            
            final_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_for_openai
            )
            bot_reply = final_response.choices[0].message.content
            
        else:
            bot_reply = ai_message.content
            
        # 🧠 Save the AI's response to this specific guest's memory history too!
        conversation_memory[guest_phone].append({"role": "assistant", "content": bot_reply})
            
    except Exception as e:
        bot_reply = f"System error! 🛠️ {str(e)}"

    # F. SEND THE TEXT BACK TO THE GUEST
    resp = MessagingResponse()
    resp.message(bot_reply)
    return str(resp)

if __name__ == '__main__':
    app.run(port=5000, debug=True)