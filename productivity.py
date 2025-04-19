import os
import sqlite3
import json
import time
import logging
import random
from config import ProductionConfig
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from google.generativeai.generative_models import GenerativeModel
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
import re
import redis
import tenacity

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize rate limiter with Redis, fall back to in-memory if Redis fails
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
    storage_uri = "redis://localhost:6379/0"
except redis.ConnectionError:
    logger.warning("Redis unavailable, falling back to in-memory storage for rate limiting")
    storage_uri = "memory://"

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"],
    storage_uri=storage_uri,
)

# Load API key securely
VIRTUZENAI_ENHANCER_KEY = os.getenv("VIRTUZENAI_ENHANCER_KEY")
if not VIRTUZENAI_ENHANCER_KEY:
    logger.error("VIRTUZENAI_ENHANCER_KEY not set in environment variables")
    raise ValueError("API key is required")

logger.debug("Configuring Gemini API...")
genai.configure(api_key=VIRTUZENAI_ENHANCER_KEY)
core_intelligence = GenerativeModel("gemini-1.5-flash")

def init_db():
    """Initialize SQLite database with required tables."""
    try:
        with sqlite3.connect("virtuzenai.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    points INTEGER DEFAULT 0,
                    preferences TEXT DEFAULT '{}'
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    description TEXT,
                    completed BOOLEAN DEFAULT 0,
                    complexity INTEGER DEFAULT 1,
                    created_at TIMESTAMP,
                    priority INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    message TEXT,
                    sentiment TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            conn.commit()
            logger.debug("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise

init_db()

# Random motivational phrases for human-like responses
MOTIVATIONAL_PHRASES = [
    "Let’s make today count!",
    "You’ve got this—let’s dive in!",
    "Ready to shine? Let’s do this!",
    "One step at a time, you’re unstoppable!",
    "Let’s turn that energy into action!"
]

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.debug(f"Retrying Gemini API call: attempt {retry_state.attempt_number}")
)
def call_gemini_api(prompt):
    """Call Gemini API with retry mechanism."""
    try:
        response = core_intelligence.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise

def enhance_user_experience(user_input, user_id):
    """Analyze user input sentiment and provide personalized productivity suggestions."""
    try:
        # Sanitize user input
        user_input = re.sub(r'[<>;]', '', user_input.strip())
        if not user_input:
            return {
                "sentiment": "neutral",
                "explanation": "Empty input provided",
                "suggestion": f"Virtuzenai suggests: Try jotting down one goal for today. {random.choice(MOTIVATIONAL_PHRASES)}"
            }

        # Fetch user data
        with sqlite3.connect("virtuzenai.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT preferences, points FROM users WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            cursor.execute("SELECT message, sentiment FROM interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5", (user_id,))
            history = cursor.fetchall()
            cursor.execute("SELECT description, complexity, priority FROM tasks WHERE user_id = ? AND completed = 0", (user_id,))
            tasks = cursor.fetchall()

        preferences = json.loads(user_data[0]) if user_data else {}
        points = user_data[1] if user_data else 0
        context = "Recent interactions: " + "; ".join([f"Message: {h[0]}, Sentiment: {h[1]}" for h in history]) if history else "No recent interactions."
        task_context = "Pending tasks: " + "; ".join([f"Task: {t[0]}, Complexity: {t[1]}, Priority: {t[2]}" for t in tasks]) if tasks else "No pending tasks."
        user_summary = f"User has {points} points, preferences: {json.dumps(preferences)}."

        # Refined sentiment analysis with nuanced emotions
        sentiment_prompt = f"""
        You are Virtuzenai, an empathetic and intelligent productivity coach. Analyze the sentiment of the user input below and classify it as one of: 'positive', 'negative', 'overwhelmed', 'neutral', 'frustrated', 'hopeful', 'anxious', 'motivated', 'tired'. Consider the following for context:
        - {context}
        - {task_context}
        - {user_summary}
        Provide a productivity suggestion that is empathetic, actionable, and tailored to the user's mood and preferences. Ensure the suggestion is concise and motivational. Return a JSON response:
        {{
            "sentiment": "mood",
            "explanation": "why this mood was inferred",
            "suggestion": "productivity tip"
        }}
        User input: "{user_input}"
        """
        logger.debug(f"Sending prompt to Gemini: {sentiment_prompt[:100]}...")
        raw_response = call_gemini_api(sentiment_prompt)
        logger.debug(f"Gemini response: {raw_response[:100]}...")

        try:
            # Remove markdown code fences if present
            raw_response = raw_response.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(raw_response)
            # Personalize suggestion based on preferences
            suggestion = data['suggestion'].capitalize()
            if preferences.get("work_style") == "short_bursts":
                suggestion = f"Try a quick {suggestion.lower()} in a 15-minute sprint."
            elif preferences.get("work_style") == "deep_focus":
                suggestion = f"Set aside 45 minutes for a focused {suggestion.lower()}."
            data["suggestion"] = f"Virtuzenai suggests: {suggestion} {random.choice(MOTIVATIONAL_PHRASES)}"
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}, raw response: {raw_response}")
            return {
                "sentiment": "neutral",
                "explanation": f"Failed to parse Gemini response: {e}",
                "suggestion": f"Virtuzenai suggests: Start with a small task to get moving. {random.choice(MOTIVATIONAL_PHRASES)}"
            }
    except Exception as e:
        logger.error(f"Error in enhance_user_experience: {e}")
        return {
            "sentiment": "neutral",
            "explanation": f"Processing error occurred: {e}",
            "suggestion": f"Virtuzenai suggests: Take a moment to breathe, then pick one small task. {random.choice(MOTIVATIONAL_PHRASES)}"
        }

def suggest_proactive_task(user_id, sentiment):
    """Suggest a proactive task based on user sentiment, time of day, and preferences."""
    try:
        with sqlite3.connect("virtuzenai.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT preferences FROM users WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            preferences = json.loads(user_data[0]) if user_data else {}
            cursor.execute("SELECT description, complexity, priority FROM tasks WHERE user_id = ? AND completed = 0", (user_id,))
            tasks = cursor.fetchall()

        hour = datetime.now().hour
        time_context = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        energy_level = "high" if time_context == "morning" and sentiment in ["positive", "motivated", "hopeful"] else "medium" if time_context == "afternoon" else "low"
        task_context = "Pending tasks: " + "; ".join([f"Task: {t[0]}, Complexity: {t[1]}, Priority: {t[2]}" for t in tasks]) if tasks else "No pending tasks."

        # Prioritize tasks with higher priority or lower complexity for low energy
        task_prompt = f"""
        As Virtuzenai, suggest a single, actionable productivity task for a user with {sentiment} mood in the {time_context} (energy level: {energy_level}). Consider:
        - {task_context}
        - Preferences: {json.dumps(preferences)}
        Prioritize tasks with higher priority or lower complexity if energy is low. Return a JSON:
        {{
            "task": "suggested task description",
            "complexity": 1-3
        }}
        """
        raw_response = call_gemini_api(task_prompt)
        try:
            raw_response = raw_response.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(raw_response)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in suggest_proactive_task: {e}, raw response: {raw_response}")
            return {"task": "Write a quick to-do list for today", "complexity": 1}
    except Exception as e:
        logger.error(f"Error in suggest_proactive_task: {e}")
        return {"task": "Write a quick to-do list for today", "complexity": 1}

def handle_origin_query(user_input):
    """Handle queries about Virtuzenai's origin with a friendly tone."""
    keywords = ["who made", "where from", "who created", "what are you", "your origin"]
    if any(keyword in user_input.lower() for keyword in keywords):
        return {
            "message": "Hey there! I’m Virtuzenai, brought to life by Innovie Solutions to help you spark your productivity with a dash of creativity!",
            "sentiment": "neutral",
            "explanation": "User asked about Virtuzenai's origin."
        }
    return None

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute")
def submit_message():
    """Handle user message submission and provide sentiment-based response."""
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        message = data.get("message")

        if not user_id or not message:
            return jsonify({"error": "user_id and message are required"}), 400

        # Validate input
        if len(message) > 1000 or re.search(r'[<>;]', message):
            return jsonify({"error": "Invalid input: message too long or contains invalid characters"}), 400
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return jsonify({"error": "Invalid user_id: use alphanumeric characters, underscores, or hyphens only"}), 400

        with sqlite3.connect("virtuzenai.db") as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO users (user_id, points) VALUES (?, 0)", (user_id,))
            conn.commit()

        origin_response = handle_origin_query(message)
        if origin_response:
            return jsonify(origin_response)

        response_data = enhance_user_experience(message, user_id)

        with sqlite3.connect("virtuzenai.db") as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO interactions (user_id, message, sentiment, timestamp) VALUES (?, ?, ?, ?)",
                           (user_id, message, response_data["sentiment"], datetime.now()))
            conn.commit()

        # Enhanced tone with empathy and variability
        tone = {
            'positive': 'Wow, your vibe is contagious! ',
            'negative': 'I hear you—let’s turn things around. ',
            'overwhelmed': 'Let’s take it one step at a time. ',
            'neutral': 'Ready to kick things off? ',
            'frustrated': 'I feel that—let’s channel it into progress. ',
            'hopeful': 'Love that optimism! Let’s build on it. ',
            'anxious': 'It’s okay to feel this way—let’s ease in. ',
            'motivated': 'You’re on fire! Let’s keep it going! ',
            'tired': 'Feeling low? Let’s start small and recharge. '
        }.get(response_data['sentiment'], 'Let’s make today awesome! ')

        proactive_task = suggest_proactive_task(user_id, response_data["sentiment"])

        return jsonify({
            "message": tone + response_data["suggestion"],
            "sentiment": response_data["sentiment"],
            "explanation": response_data["explanation"],
            "proactive_task": proactive_task
        })
    except Exception as e:
        logger.error(f"Error in submit_message: {e}")
        return jsonify({"error": "Oops, something went wrong. Let’s try again!"}), 500

@app.route("/add_task", methods=["POST"])
@limiter.limit("10 per minute")
def add_task():
    """Add a new task for a user with priority support."""
    try:
        logger.debug("Received add_task request")
        data = request.get_json()
        logger.debug(f"Request data: {data}")
        user_id = data.get("user_id")
        description = data.get("description")
        complexity = min(max(data.get("complexity", 1), 1), 3)
        priority = min(max(data.get("priority", 0), 0), 5)  # New priority field

        if not user_id or not description:
            logger.error("Missing user_id or description")
            return jsonify({"error": "user_id and description are required"}), 400

        # Validate input
        logger.debug("Validating input")
        if len(description) > 500 or re.search(r'[<>;]', description):
            logger.error("Invalid description: too long or contains invalid characters")
            return jsonify({"error": "Invalid description: too long or contains invalid characters"}), 400
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            logger.error("Invalid user_id: use alphanumeric characters, underscores, or hyphens only")
            return jsonify({"error": "Invalid user_id: use alphanumeric characters, underscores, or hyphens only"}), 400

        logger.debug("Connecting to database")
        with sqlite3.connect("virtuzenai.db") as conn:
            cursor = conn.cursor()
            logger.debug("Executing INSERT query")
            cursor.execute(
                "INSERT INTO tasks (user_id, description, complexity, created_at, priority) VALUES (?, ?, ?, ?, ?)",
                (user_id, description, complexity, datetime.now(), priority)
            )
            conn.commit()
            logger.debug("Task inserted successfully")

        return jsonify({"message": f"Task added! {random.choice(MOTIVATIONAL_PHRASES)}"})
    except Exception as e:
        logger.error(f"Error in add_task: {str(e)}", exc_info=True)
        return jsonify({"error": "Oops, something went wrong. Let’s try again!"}), 500

@app.route("/complete_task", methods=["POST"])
@limiter.limit("10 per minute")
def complete_task():
    """Mark a task as completed and update user points."""
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        task_id = data.get("task_id")

        if not user_id or not task_id:
            return jsonify({"error": "user_id and task_id are required"}), 400

        # Validate input
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return jsonify({"error": "Invalid user_id: use alphanumeric characters, underscores, or hyphens only"}), 400
        try:
            task_id = int(task_id)
        except ValueError:
            return jsonify({"error": "Invalid task_id: must be an integer"}), 400

        with sqlite3.connect("virtuzenai.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT completed, complexity, priority FROM tasks WHERE task_id = ? AND user_id = ?", (task_id, user_id))
            task = cursor.fetchone()

            if not task:
                return jsonify({"error": "Task not found"}), 404

            if task[0]:
                return jsonify({"error": "Task already completed"}), 400

            cursor.execute("SELECT sentiment FROM interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
            last_sentiment = cursor.fetchone()
            sentiment = last_sentiment[0] if last_sentiment else "neutral"
            points = task[1] * 10
            if sentiment in ["negative", "overwhelmed", "frustrated", "anxious", "tired"]:
                points *= 1.5
            if task[2] >= 3:  # Bonus for high-priority tasks
                points *= 1.2

            cursor.execute("UPDATE tasks SET completed = 1 WHERE task_id = ?", (task_id,))
            cursor.execute("UPDATE users SET points = points + ? WHERE user_id = ?", (int(points), user_id))
            conn.commit()

            cursor.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
            total_points = cursor.fetchone()[0]
            badge = None
            if total_points >= 100 and total_points < 200:
                badge = "Focus Spark: You’re lighting up your productivity!"
            elif total_points >= 200 and total_points < 500:
                badge = "Productivity Master: You’re absolutely crushing it!"
            elif total_points >= 500:
                badge = "Virtuzenai Legend: Your focus is inspiring!"

        response = {"message": f"Nice work! You earned {int(points)} points for completing that task!"}
        if badge:
            response["badge"] = badge
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in complete_task: {e}")
        return jsonify({"error": "Oops, something went wrong. Let’s try again!"}), 500

@app.route("/user_progress", methods=["GET"])
@limiter.limit("20 per minute")
def user_progress():
    """Retrieve user progress, including points and tasks."""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        # Validate input
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return jsonify({"error": "Invalid user_id: use alphanumeric characters, underscores, or hyphens only"}), 400

        with sqlite3.connect("virtuzenai.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT points, preferences FROM users WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()
            cursor.execute("SELECT task_id, description, completed, complexity, priority FROM tasks WHERE user_id = ?", (user_id,))
            tasks = cursor.fetchall()

        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "user_id": user_id,
            "points": user[0],
            "preferences": json.loads(user[1]),
            "tasks": [{"task_id": t[0], "description": t[1], "completed": bool(t[2]), "complexity": t[3], "priority": t[4]} for t in tasks]
        })
    except Exception as e:
        logger.error(f"Error in user_progress: {e}")
        return jsonify({"error": "Oops, something went wrong. Let’s try again!"}), 500

@app.route("/update_preferences", methods=["POST"])
@limiter.limit("5 per minute")
def update_preferences():
    """Update user preferences."""
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        preferences = data.get("preferences", {})

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        # Validate input
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return jsonify({"error": "Invalid user_id: use alphanumeric characters, underscores, or hyphens only"}), 400
        if not isinstance(preferences, dict):
            return jsonify({"error": "Preferences must be a valid JSON object"}), 400

        with sqlite3.connect("virtuzenai.db") as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET preferences = ? WHERE user_id = ?", (json.dumps(preferences), user_id))
            conn.commit()

        return jsonify({"message": f"Preferences updated! I’m now even more in tune with you! {random.choice(MOTIVATIONAL_PHRASES)}"})
    except Exception as e:
        logger.error(f"Error in update_preferences: {e}")
        return jsonify({"error": "Oops, something went wrong. Let’s try again!"}), 500

if __name__ == "__main__":
    app.config.from_object(DevelopmentConfig)
    logger.info("Starting Flask server in development mode on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
else:
    app.config.from_object(ProductionConfig)
    logger.info("Starting Flask server in production mode")
