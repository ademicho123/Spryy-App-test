import os
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from translator import translate_text, real_time_translate_audio, user_languages
import daily
from daily import CallClient

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Get API keys from environment variables
DAILY_API_KEY = os.getenv('DAILY_API_KEY')
DAILY_API_URL = 'https://api.daily.co/v1/rooms'

# Initialize Daily context
daily.daily_core_context_create()

# Initialize Daily client
daily_client = CallClient()

# Helper function to create a room using the REST API
def create_daily_room(properties=None):
    headers = {
        'Authorization': f'Bearer {DAILY_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = properties if properties else {}
    response = requests.post(DAILY_API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/create-instant-meeting', methods=['POST'])
def create_instant_meeting():
    try:
        room = create_daily_room()
        return jsonify({'id': room['name'], 'url': room['url']})
    except Exception as e:
        logging.error(f"Failed to create instant meeting room: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/schedule-meeting', methods=['GET'])
def schedule_meeting_page():
    return render_template('schedule_meeting.html')

@app.route('/create-scheduled-meeting', methods=['POST'])
def create_scheduled_meeting():
    data = request.json
    meeting_time = datetime.fromisoformat(data.get('time'))
    expiration_time = meeting_time + timedelta(hours=2)  # Set expiration 2 hours after meeting start
    try:
        room = create_daily_room(properties={
            'exp': int(expiration_time.timestamp()),
            'enable_knocking': False,
            'start_audio_off': True,
            'start_video_off': True
        })
        return jsonify({'id': room['name'], 'url': room['url']})
    except Exception as e:
        logging.error(f"Failed to create scheduled meeting room: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/meeting_language/<meeting_id>', methods=['GET'])
def meeting_language_selection(meeting_id):
    return render_template('meeting_lang.html', meeting_id=meeting_id)

@app.route('/meeting_room/<room_id>', methods=['GET'])
def meeting_room(room_id):
    language = request.args.get('language', 'en')  # Default to English if not provided
    return render_template('meeting_room.html', room_id=room_id, language=language)

@app.route('/get-room-url/<room_id>', methods=['GET'])
def get_room_url(room_id):
    headers = {
        'Authorization': f'Bearer {DAILY_API_KEY}',
        'Content-Type': 'application/json'
    }
    response = requests.get(f'{DAILY_API_URL}/{room_id}', headers=headers)
    if response.status_code == 200:
        room_details = response.json()
        return jsonify({'url': room_details['url']})
    else:
        logging.error(f"Failed to get room URL: {response.status_code}, {response.text}")
        return jsonify(response.json()), response.status_code

@socketio.on('set_language')
def handle_set_language(data):
    user_id = request.sid  # Unique session ID for the connected user
    user_languages[user_id] = data['language']
    logging.info(f"User {user_id} set language to {data['language']}")

@socketio.on('start_translation')
def handle_start_translation(data):
    user_id = data['user_id']
    from_language = data['from_language']
    to_language = data['to_language']
    real_time_translate_audio(from_language, to_language, user_id)

@socketio.on('translate')
def handle_translate(data):
    text = data['text']
    from_language = data['from_language']
    to_language = data['to_language']
    translated_text = translate_text(text, to_language)
    emit('translated_chat_message', {'text': translated_text}, room=request.sid)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)