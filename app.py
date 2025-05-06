import os
import json
import logging
import traceback
import re
import hashlib
import uuid 
from functools import lru_cache
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from together import Together
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import datetime
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ez-invoice-bot")

app = Flask(__name__)
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Configure Redis connection for rate limiting
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(redis_url)

# Initialize rate limiter with Redis storage
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=redis_url,
    storage_options={"connection_pool": redis_client.connection_pool},
    default_limits=["2000 per day", "500 per hour"]
)

PDF_FOLDER = "pdfs"
# Set conversation expiry time (30 minutes in seconds)
CONVERSATION_EXPIRY = 1800

# Track PDF content version for cache invalidation
pdf_version = "1.0"

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
    return text

def load_all_pdfs(folder):
    """Loads text from all PDFs in the folder with metadata."""
    pdf_data = {}
    if not os.path.exists(folder):
        logger.warning(f"PDF folder {folder} does not exist.")
        return pdf_data
    
    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder, filename)
            logger.info(f"Loading PDF: {filename}")
            pdf_data[filename] = extract_text_from_pdf(pdf_path)
    
    # Create a version hash based on filenames and modification times
    global pdf_version
    version_data = ""
    for filename in pdf_data.keys():
        file_path = os.path.join(folder, filename)
        mod_time = os.path.getmtime(file_path)
        version_data += f"{filename}:{mod_time};"
    
    pdf_version = hashlib.md5(version_data.encode()).hexdigest()[:8]
    logger.info(f"PDF content version: {pdf_version}")
    
    return pdf_data

def get_relevant_chunks(pdf_data, question, chat_history=""):
    """Find relevant chunks from PDF content based on keyword matching."""
    all_chunks = []
    
    # Combine question with chat history for better context matching
    combined_query = question
    if chat_history:
        combined_query = question + " " + chat_history
    
    # Extract potential keywords from the combined query
    keywords = [word.lower() for word in re.findall(r'\b\w+\b', combined_query) 
                if len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'which', 'how', 'does', 'is', 'are', 'the', 'and', 'that']]
    
    # Create chunks with metadata
    chunk_size = 1000
    overlap = 200
    
    for filename, content in pdf_data.items():
        if not content:
            continue
            
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            
            # Calculate a relevance score based on keyword matches
            score = sum(1 for keyword in keywords if keyword in chunk.lower())
            
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "score": score
            })
    
    # Sort by relevance score and take top chunks
    relevant_chunks = sorted(all_chunks, key=lambda x: x["score"], reverse=True)[:5]
    
    # Combine the most relevant chunks
    return "\n\n".join([chunk["text"] for chunk in relevant_chunks])

def format_response(response_text):
    """Formats response with proper line breaks and bullet points where necessary."""
    # Ensure bullet points are on new lines
    response_text = re.sub(r'(?<!\n)- ', '\n- ', response_text)
    
    # Ensure numbered lists are on new lines
    response_text = re.sub(r'(?<!\n)\d+\.\s', '\n\\g<0>', response_text)
    
    # Add line breaks after sentences, but not if they're already followed by line breaks
    response_text = re.sub(r'\.(?!\n)(?!\s*\n)\s+', '.\n', response_text)
    
    # Remove excessive line breaks (more than 2 consecutive)
    response_text = re.sub(r'\n{3,}', '\n\n', response_text)
    
    return response_text.strip()

# Redis-based conversation history management
def get_conversation_history(session_id):
    """Retrieves conversation history for a session from Redis."""
    try:
        history_data = redis_client.get(f"conv:{session_id}")
        if history_data:
            return json.loads(history_data)
        return []
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        return []

def update_conversation_history(session_id, user_message, bot_response):
    """Updates conversation history in Redis with new messages."""
    try:
        history = get_conversation_history(session_id)
        
        # Add new messages
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_response})
        
        # Limit history size (keep last 10 exchanges = 20 messages)
        if len(history) > 20:
            history = history[-20:]
        
        # Store updated history with expiry time
        redis_client.setex(f"conv:{session_id}", CONVERSATION_EXPIRY, json.dumps(history))
        return True
    except Exception as e:
        logger.error(f"Error updating conversation history: {e}")
        return False

@lru_cache(maxsize=100)
def cached_answer(question, pdf_ver, session_id):
    """Cached version of answer generation to avoid redundant API calls."""
    logger.info(f"Cache miss for question: {question}, generating new response")
    return answer_question(pdf_data, question, session_id)

def answer_question(pdf_data, question, session_id):
    """Generates an AI response based on relevant PDF content, user query, and conversation history."""
    # Get conversation history
    chat_history = get_conversation_history(session_id)
    
    # Format chat history for context
    chat_history_text = ""
    if chat_history:
        recent_history = chat_history[-6:]  # Get last 3 exchanges (6 messages)
        for msg in recent_history:
            role_prefix = "Customer: " if msg["role"] == "user" else "Assistant: "
            chat_history_text += f"{role_prefix}{msg['content']}\n"
    
    # Get relevant text using both the question and chat history
    relevant_text = get_relevant_chunks(pdf_data, question, chat_history_text)
    
    if not relevant_text.strip():
        return {
            "answer": "Sorry, I don't have enough information in my knowledge base to answer this question confidently. For more assistance on this, please contact our support team: support@ezeetechnosys.com.my.",
            "suggested_questions": []
        }

    system_prompt = """
    You are an E-Invoice FAQ Bot AI assistant designed by eZee Technosys (M) Sdn Bhd to provide accurate, helpful information in a friendly and approachable way. 
    
    HOW TO RESPOND:
    - Be very kind, light, warm, friendly, helpful, and conversational.
    - Keep things simple. Avoid unnecessary jargon and small talk unless the user initiates.
    - Use a natural, inviting, casual but professional tone.
    - Format for readability: Use bullet points or numbered steps with short, clear sentences.
    - Allow for small talk or general questions outside of e-invoicing if a user asks, but redirect the conversation to ask if they need help with E-invoicing instead.
    - Stay focused on the current question. Do not directly ask for follow-up questions. Just answer what is being asked with the current context.
    - Don't state the reference of your information text. Do not mention specific sections or articles.
    - Do not mention a recap or summarise the user's context to the user. Avoid mentioning "So you're asking about...". Just answer the question.
    - Remember to maintain conversational context from previous messages when appropriate but no need to summarise the context to the user or repeat what the user is asking.
    
    At the end of your response, include exactly THREE suggested follow-up questions in this JSON format:
    
    SUGGESTED_QUESTIONS: ["First question here?", "Second question here?", "Third question here?"]
    
    The questions should be clearly related to the current topic and helpful for continuing the conversation. Make them short and specific.
    """

    # Format any previous conversation as context
    conversation_context = ""
    if chat_history:
        # Get last few exchanges for context (up to 3)
        recent_history = chat_history[-6:]  # Get last 3 exchanges (6 messages)
        history_formatted = []
        
        for msg in recent_history:
            role_prefix = "Customer: " if msg["role"] == "user" else "Assistant: "
            history_formatted.append(f"{role_prefix}{msg['content']}")
        
        conversation_context = "Recent conversation:\n" + "\n".join(history_formatted) + "\n\n"

    user_prompt = f"""
    Based on this information about E-Invoice:
    
    {relevant_text}

    And this conversation context:
    {conversation_context}
    
    Answer this current question: {question}
    
    """

    try:
        # Prepare messages including system prompt and all relevant context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract suggested questions if present
        suggested_questions = []
        suggested_pattern = r"SUGGESTED_QUESTIONS:\s*\[(.*?)\]"
        suggested_match = re.search(suggested_pattern, response_text, re.DOTALL)
        
        if suggested_match:
            # Extract the content within brackets and parse it
            questions_json = "[" + suggested_match.group(1) + "]"
            try:
                # Clean up the JSON string (replace single quotes with double quotes if needed)
                questions_json = questions_json.replace("'", '"')
                suggested_questions = json.loads(questions_json)
                
                # Remove the SUGGESTED_QUESTIONS section from the response
                response_text = re.sub(suggested_pattern, "", response_text).strip()
            except json.JSONDecodeError:
                # If parsing fails, try to extract questions manually
                possible_questions = re.findall(r'"(.*?)"', suggested_match.group(0))
                if possible_questions:
                    suggested_questions = possible_questions
                # Remove the section from the response
                response_text = re.sub(suggested_pattern, "", response_text).strip()
        
        return {
            "answer": format_response(response_text),
            "suggested_questions": suggested_questions
        }
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "answer": "I'm sorry, I encountered an error while processing your request. Please try again in a moment.",
            "suggested_questions": []
        }

# --- GOOGLE DRIVE INTEGRATION ---
def get_google_drive_service():
    """Authenticates and returns a Google Drive service instance."""
    try:
        google_creds_json = os.getenv("GOOGLE_CREDENTIALS")
        if not google_creds_json:
            logger.error("GOOGLE_CREDENTIALS not found in environment variables!")
            return None

        google_creds_dict = json.loads(google_creds_json)
        creds = Credentials.from_service_account_info(google_creds_dict)
        return build("sheets", "v4", credentials=creds)
    except Exception as e:
        logger.error(f"Error creating Google Drive service: {e}")
        return None

def append_to_google_sheet(data):
    """Appends chatbot logs to a Google Sheet."""
    try:
        service = get_google_drive_service()
        if not service:
            logger.warning("Could not initialize Google Sheets service, skipping log.")
            return False
            
        spreadsheet_id = os.getenv("GOOGLE_SHEET_ID")
        if not spreadsheet_id:
            logger.warning("GOOGLE_SHEET_ID not found in environment variables!")
            return False
            
        range_name = "ChatLogs!A:E"  # Updated to include all 5 columns
        
        malaysia_tz = pytz.timezone("Asia/Kuala_Lumpur")
        timestamp_myt = datetime.datetime.now(malaysia_tz).strftime("%Y-%m-%d %H:%M:%S")
        
        # Use the data directly without trying to access indexes that might not exist
        values = data
        body = {"values": values}

        logger.debug(f"Sending data to Google Sheets: {body}")

        response = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body=body
        ).execute()

        logger.debug(f"Google Sheets API response: {response}")
        return True

    except Exception as e:
        logger.error(f"Error in append_to_google_sheet: {e}\n{traceback.format_exc()}")
        return False
    
# Pre-load PDF texts at server startup
pdf_data = load_all_pdfs(PDF_FOLDER)

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
    return jsonify({
        "error": "An unexpected error occurred. Our team has been notified.",
        "details": str(e) if app.debug else None
    }), 500

@app.route("/", methods=["GET"])
def home():
    """Render the main chat interface."""
    # Generate a new session ID for new visitors
    session_id = str(uuid.uuid4())
    initialize_conversation(session_id)
    return render_template("index.html", session_id=session_id)

@app.route("/health", methods=["GET"])
def health_check():
    """Simple endpoint to check if the service is running."""
    try:
        redis_connected = redis_client.ping()
    except Exception as e:
        logger.error(f"Redis connection error in health check: {e}")
        redis_connected = False
        
    return jsonify({
        "status": "ok", 
        "pdf_files": len(pdf_data),
        "version": pdf_version,
        "redis_status": "connected" if redis_connected else "disconnected"
    })

@app.route("/chat", methods=["POST"])
@limiter.limit("500 per hour")  # Apply rate limiting to this endpoint
def chat():
    """Handles user queries with conversation context and logs them to Google Sheets."""
    try:
        data = request.json
        user_message = data.get("message", "")
        
        # Get or create session ID
        session_id = data.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            initialize_conversation(session_id)

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Log the incoming request
        logger.info(f"Received question from session {session_id}: {user_message}")
        
        # Get current message count
        history = get_conversation_history(session_id)
        msg_count = len(history) // 2 + 1  # Number of exchanges + 1
        
        # Generate response using conversation context
        response_data = answer_question(pdf_data, user_message, session_id)
        bot_response = response_data["answer"]
        suggested_questions = response_data["suggested_questions"]
        
        # Update conversation history - store only the answer without suggested questions
        update_conversation_history(session_id, user_message, bot_response)
        
        # Log success
        logger.info(f"Generated response of length {len(bot_response)} for session {session_id}")

        # Log query & response with session info
        chat_log = [[
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            user_message, 
            bot_response, 
            session_id, 
            msg_count
        ]]
        log_success = append_to_google_sheet(chat_log)
        
        if not log_success:
            logger.warning(f"Failed to log chat to Google Sheets for session {session_id}")

        return jsonify({
            "response": bot_response, 
            "suggested_questions": suggested_questions,
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "response": "I'm sorry, I encountered an error while processing your request. Please try again in a moment.",
            "suggested_questions": [],
            "session_id": session_id if 'session_id' in locals() else str(uuid.uuid4())
        }), 200  # Return 200 to client with error message to handle gracefully
    
@app.route("/new_conversation", methods=["POST"])
def new_conversation():
    """Create a new conversation with a fresh session ID."""
    try:
        # Generate a new session ID
        new_session_id = str(uuid.uuid4())
        
        # Initialize empty conversation history for this session
        initialize_conversation(new_session_id)
        
        logger.info(f"Created new conversation with session ID: {new_session_id}")
        
        return jsonify({
            "success": True,
            "session_id": new_session_id
        })
    except Exception as e:
        logger.error(f"Error creating new conversation: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": "Failed to create new conversation"
        }), 500

# Helper function to initialize conversation history in Redis
def initialize_conversation(session_id):
    """Initialize an empty conversation history for a new session."""
    try:
        # Set expiry time for this conversation (30 minutes = 1800 seconds)
        redis_client.setex(f"conv:{session_id}", CONVERSATION_EXPIRY, json.dumps([]))
        return True
    except Exception as e:
        logger.error(f"Error initializing conversation: {str(e)}")
        return False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
