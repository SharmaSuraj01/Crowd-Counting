# 🚀 AI Crowd Counting System v2.0

Advanced real-time crowd and vehicle detection system using YOLOv8 neural network with professional architecture, database persistence, and comprehensive logging.

## ✨ Features

### Core Features
- 👥 **Real-time Crowd Detection** - Detect and count people in images/videos
- 🚗 **Vehicle Detection** - Identify and count cars
- 🔥 **Heatmap Generation** - Visualize crowd density distribution
- 📊 **Analytics Dashboard** - View statistics and analysis history
- ⚠️ **Alert System** - Configurable thresholds with real-time alerts
- 📈 **Video Analysis** - Frame-by-frame crowd count tracking with charts

### Technical Improvements
- 🗄️ **SQLite Database** - Persistent storage of all analyses
- 📝 **Comprehensive Logging** - Rotating log files with multiple levels
- ⚙️ **Environment Configuration** - Easy setup via .env file
- 🛡️ **Error Handling** - Robust exception handling with user-friendly messages
- 📡 **REST API** - Clean API with `/api/*` endpoints
- 🎨 **Modern UI** - Beautiful responsive interface with Tailwind CSS
- 📦 **Production Ready** - Proper separation of concerns and best practices

## 📋 System Architecture

```
Crowd-Counting/
├── main.py                    # FastAPI backend
├── config.py                  # Configuration management
├── database.py                # SQLite database handler
├── logger.py                  # Logging setup
├── server.js                  # Express.js frontend server
├── package.json               # Node.js dependencies
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore file
├── views/
│   └── index.ejs              # Frontend UI
├── logs/
│   └── app.log                # Application logs
└── temp/                      # Temporary video files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- pip (Python package manager)
- npm (Node package manager)

### Installation

#### 1. Clone/Setup Project
```bash
cd "Crowd-Counting"
```

#### 2. Setup Python Backend

```bash
# Create Python virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate  # On Windows

# Install Python dependencies
pip install -r requirements.txt
```

#### 3. Setup Environment

```bash
# Copy example env file and customize if needed
copy .env.example .env

# Or on Linux/Mac:
cp .env.example .env

# Edit .env if you want to change port numbers or settings
```

#### 4. Setup Node.js Frontend

```bash
# Install Node dependencies
npm install
```

### Running the Application

#### Terminal 1 - Start Python Backend
```bash
python main.py
```
You should see output like:
```
INFO: Started server process [12345]
INFO: Uvicorn running on http://127.0.0.1:8001
```

#### Terminal 2 - Start Node.js Frontend
```bash
npm start
```
You should see output like:
```
Server running on http://localhost:3000
Python API URL: http://127.0.0.1:8001
```

#### Open in Browser
Navigate to: http://localhost:3000

## 📚 API Endpoints

All endpoints are available at `http://localhost:8001`

### Health Check
- **GET** `/health` - API health status

### Image Analysis
- **POST** `/upload?threshold=10` - Upload and analyze image
  - Form data: `file` (image file)
  - Returns: detection count, heatmap, confidence scores

### Video Analysis
- **POST** `/upload-video` - Upload and analyze video
  - Form data: `file` (video file)
  - Returns: frame-by-frame analysis, charts data

### Webcam
- **GET** `/webcam-frame?threshold=10` - Capture and analyze webcam frame

### History & Statistics
- **GET** `/history?limit=20` - Get recent analyses
- **GET** `/stats` - Get overall statistics
- **DELETE** `/history` - Clear all history

## ⚙️ Configuration

Edit `.env` file to customize:

```ini
# API Settings
API_HOST=127.0.0.1
API_PORT=8001

# Frontend
FRONTEND_PORT=3000
PYTHON_API_URL=http://127.0.0.1:8001

# Database
DATABASE_PATH=crowd_counting.db

# Alert threshold
DEFAULT_ALERT_THRESHOLD=10

# File limits
MAX_IMAGE_SIZE=50000000        # 50MB
MAX_VIDEO_SIZE=500000000       # 500MB

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Model
MODEL_PATH=yolov8n.pt
```

## 📊 Database

### Tables
- **analysis** - Stores all detection results
- **video_frames** - Stores frame-by-frame video analysis
- **settings_store** - Configuration storage

### Viewing Database
```bash
# Install sqlite3
pip install sqlite3

# Open database
sqlite3 crowd_counting.db

# List tables
.tables

# View analysis records
SELECT * FROM analysis;
```

## 📝 Logging

Logs are saved to `logs/app.log` with automatic rotation.

Log levels: INFO, WARNING, ERROR

View logs:
```bash
# Windows
type logs\app.log

# Linux/Mac
cat logs/app.log

# Follow logs in real-time (Linux/Mac)
tail -f logs/app.log
```

## 🔍 Troubleshooting

### Issue: "Cannot connect to API"
- Ensure Python backend is running on terminal 1
- Check if port 8001 is not in use: `lsof -i :8001`
- Check firewall settings

### Issue: Model fails to load
- Ensure `yolov8n.pt` is in the project root
- Download it: `from ultralytics import YOLO; YOLO('yolov8n.pt')`

### Issue: Video processing is slow
- This is normal for first run (model loading)
- Large videos will take time proportional to their length
- Processing time is shown in results

### Issue: Database locked error
- Ensure only one instance of the app is running
- Delete `crowd_counting.db` and restart (loses history)

### Issue: Port already in use
Edit `.env` to use different ports:
```ini
API_PORT=8002
FRONTEND_PORT=3001
```

## 🎓 College Project Features

This project demonstrates:

1. **Backend Architecture**
   - REST API design with FastAPI
   - Proper error handling
   - Database persistence
   - Logging and monitoring

2. **Frontend Development**
   - Modern responsive UI
   - Real-time status updates
   - File upload handling
   - Data visualization (Charts.js)

3. **Machine Learning**
   - YOLOv8 object detection
   - Confidence scoring
   - Batch processing

4. **DevOps Best Practices**
   - Environment configuration
   - Logging system
   - Database management
   - Code organization

5. **Professional Development**
   - Clean code architecture
   - CORS handling
   - API documentation
   - Error recovery

## 📈 Performance Tips

- Images: ~200-500ms processing time (depending on size)
- Videos: ~1-2 seconds per frame average
- First run: ~10 seconds (model loading)

## 🔐 Security Notes

- No authentication implemented (add if deploying publicly)
- File uploads limited to prevent abuse
- API has CORS enabled for local development

## 📦 Deployment

To deploy to production:

1. Set `NODE_ENV=production`
2. Use production-grade server (Gunicorn for Python, PM2 for Node)
3. Add authentication layer
4. Use HTTPS
5. Set up reverse proxy (Nginx)
6. Configure database backups

## 🐛 Known Limitations

- Single-user application
- No real-time streaming (batch processing only)
- In-memory threshold (resets on restart - uses database for persistence)
- Depends on model accuracy (YOLOv8 small model)

## 🤝 Contributing

To improve the project:

1. Add user authentication
2. Implement real-time streaming with WebSockets
3. Add more detection models
4. Optimize video processing
5. Add export features (CSV, PDF reports)
6. Deploy to cloud platforms

## 📄 License

College Project - Educational Use Only

## 👨‍💼 Project Info

- **Version**: 2.0.0
- **Created**: 2024
- **Stack**: FastAPI + Express.js + YOLOv8
- **Status**: Production Ready

## 📞 Support

For issues or questions:
1. Check logs in `logs/app.log`
2. Review error messages in browser console (F12)
3. Ensure both services are running
4. Check .env configuration

---

**Happy Crowd Counting! 🎉**
