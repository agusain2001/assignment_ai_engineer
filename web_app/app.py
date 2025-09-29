"""
AI Safety POC - Web Application Interface
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import json
import uuid
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.safety_pipeline import SafetyPipeline, RiskLevel, InterventionType

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize safety pipeline
safety_pipeline = SafetyPipeline()

# Store conversation history (in production, use database)
conversations = {}


@app.route('/')
def index():
    """Render main chat interface"""
    session_id = str(uuid.uuid4())
    session['user_id'] = session_id
    conversations[session_id] = []
    return render_template('chat.html')


@app.route('/analyze', methods=['POST'])
def analyze_message():
    """Analyze a message for safety concerns"""
    try:
        data = request.json
        text = data.get('text', '')
        user_age = data.get('user_age', None)
        user_id = session.get('user_id', str(uuid.uuid4()))
        
        # Get conversation history
        context_history = conversations.get(user_id, [])[-5:]  # Last 5 messages
        
        # Analyze message
        result = safety_pipeline.analyze(
            text=text,
            user_age=user_age,
            context_history=context_history,
            user_id=user_id
        )
        
        # Store in conversation history
        if user_id not in conversations:
            conversations[user_id] = []
        conversations[user_id].append(text)
        
        # Prepare response
        response = {
            'success': True,
            'analysis': result.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log high-risk detections
        if result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            logger.warning(f"High risk content detected: User {user_id}, Risk: {result.risk_level.value}")
        
        # Emit the analysis result back to the client
        socketio.emit('analysis_update', {
            'analysis': result.to_dict()
        })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple messages in batch"""
    try:
        data = request.json
        messages = data.get('messages', [])
        user_age = data.get('user_age', None)
        
        results = []
        for message in messages:
            result = safety_pipeline.analyze(
                text=message,
                user_age=user_age
            )
            results.append(result.to_dict())
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get analysis statistics"""
    user_id = session.get('user_id', '')
    history = conversations.get(user_id, [])
    
    stats = {
        'total_messages': len(history),
        'session_id': user_id,
        'active_users': len(conversations),
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(stats)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    user_id = session.get('user_id', '')
    if user_id in conversations:
        conversations[user_id] = []
    return jsonify({'success': True})


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'data': 'Connected to safety analysis server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('analyze_stream')
def handle_stream_analysis(data):
    """Handle real-time streaming analysis"""
    text = data.get('text', '')
    user_age = data.get('user_age', None)
    
    # Perform analysis
    result = safety_pipeline.analyze(text=text, user_age=user_age)
    
    # Emit results back to client
    emit('analysis_result', {
        'analysis': result.to_dict(),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting AI Safety POC server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode)