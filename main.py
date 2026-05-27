import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

import json
import requests

def search_google_maps(query, location="Poblenou, Barcelona, Spain"):
    """Pings the Google Places API to find live local recommendations."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    # 1. TAPE THE KEY DIRECTLY TO THE URL:
    url = f"https://places.googleapis.com/v1/places:searchText?key={api_key}"
    search_text = f"{query} near {location}"
    
    # 2. REMOVE THE KEY FROM THE HEADERS ENTIRELY:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating"
    }
    
    payload = {"textQuery": search_text}
    
    # 3. THE ULTIMATE PROOF (Prints to your terminal):
    print(f"🚨 SENDING REQUEST TO: {url}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        
        # 1. If it works perfectly:
        if "places" in data:
            results = []
            for place in data["places"][:3]: 
                name = place.get("displayName", {}).get("text", "Unknown")
                address = place.get("formattedAddress", "No address")
                rating = place.get("rating", "No rating")
                results.append(f"- **{name}** ({rating}⭐): {address}")
            return "\n".join(results)
            
        # 2. IF GOOGLE SENDS AN ERROR, SHOW IT TO US:
        elif "error" in data:
            return f"**Google API Error:** {data['error'].get('message', 'Unknown error')}"
            
        # 3. If it's just completely empty:
        else:
            return f"Google returned no places. Raw data: {data}"
            
    except Exception as e:
        return f"Error connecting to Maps: {e}"


# ==========================================
# APP UI SETUP 
# ==========================================
# 1. The Browser Tab & Top Left Logo
st.set_page_config(page_title="Sukasa Host", page_icon="logo.png")
st.logo("logo.png") 

# 2. The Main Page Banner 
col1, col2 = st.columns([1.5, 8]) # We widened the first column slightly
with col1:
    st.image("logo.png", width=130) # More than doubled the size!
with col2:
    st.title("Your Digital Host")

st.markdown("""
👋 **Bon dia! Mi casa es Sukasa.**

I'm your digital host and local guide. Ask me anything, in any language:

🏠 How the apartment works (Wi-Fi, rules, etc.)

💡 The best local tips and hidden gems

🗺️ How to get around the neighborhood

**So, what can I help you with first?**
""")
# 2. Unlock the Vaults
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("coliving-rules") # Your new database bucket

# 3. Read the Core Personality (Only once)
@st.cache_data
def load_personality():
    with open("system_prompt.txt", "r", encoding="utf-8") as file:
        return file.read()

# 4. Set up the Web Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": load_personality()}
    ]

# 5. Display previous chat messages on the screen
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 6. The Tool Menu for the AI Agent (Step 2)
agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_google_maps",
            # 👇 THIS IS THE EXACT LINE YOU ARE CHANGING:
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

# 7. The Autonomous Chat Engine (Step 3)
if user_input := st.chat_input("Ask the Concierge a question..."):
    
    # Display and save user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # --- RAG PIPELINE (Check house rules!) ---
            raw_query_embedding = openai_client.embeddings.create(
                input=user_input,
                model="text-embedding-3-small"
            ).data[0].embedding
            
            search_results = index.query(
                vector=raw_query_embedding,
                top_k=3,
                include_metadata=True
            )
            retrieved_context = "\n\n".join([match["metadata"]["text"] for match in search_results["matches"]])
            
            # Secretly inject the house rules into the AI's brain
            messages_for_openai = st.session_state.messages.copy()
            messages_for_openai.append({
                "role": "system", 
                "content": f"Here are the property rules: \n{retrieved_context}\n\nIf the user asks about property rules, use ONLY the text above. If the user asks for local stores, restaurants, or businesses, use your Google Maps tool to find them."
            })
            
            # --- THE NEW AGENTIC LOOP ---
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_for_openai,
                tools=agent_tools,
                tool_choice="auto" 
            )
            
            ai_message = response.choices[0].message
            
            # Did the AI decide to use Google Maps?
            if ai_message.tool_calls:
                
                # 👇 WE JUST ADDED THIS LINE HERE, ABOVE THE LOOP!
                messages_for_openai.append(ai_message)
                
                for tool_call in ai_message.tool_calls:
                    if tool_call.function.name == "search_google_maps":
                        
                        arguments = json.loads(tool_call.function.arguments)
                        search_query = arguments.get("query")
                        
                        st.info(f"📍 Checking live map for: {search_query}...")
                        maps_results = search_google_maps(search_query)

                        #st.error(f"🔧 DEVELOPER DEBUG MODE: {maps_results}")
                        
                        # 👇 (Notice the broken line is no longer down here!)
                        messages_for_openai.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": maps_results
                        })
                
                # Second Call: Let the AI write the final answer using the live Maps data
                final_response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_for_openai
                )
                final_text = final_response.choices[0].message.content
                st.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                
            else:
                # The AI didn't need Google Maps (it just used the Pinecone house rules)
                st.markdown(ai_message.content)
                st.session_state.messages.append({"role": "assistant", "content": ai_message.content})