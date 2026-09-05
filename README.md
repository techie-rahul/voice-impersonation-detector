# AI Voice Impersonation Detector

AI-powered system for real-time detection and prevention of voice-cloning impersonation attacks.

**SIH26104 — AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**

## Problem

AI voice-cloning technology can be used to impersonate legitimate individuals during sensitive voice interactions.

Our system analyzes incoming voice audio and combines multiple signals to determine whether an interaction is potentially fraudulent.

## MVP

The MVP focuses on:

* AI-generated / voice-cloned speech detection
* Speaker verification
* Context-aware risk analysis
* Dynamic risk scoring
* ALLOW / VERIFY / BLOCK decisions
* Prevention of high-risk actions
* Real-time risk dashboard

## System Flow

```text
Voice / Audio
      ↓
Audio Preprocessing
      ↓
AASIST Deepfake Detection
      ↓
Speaker Verification
      ↓
Context Analysis
      ↓
Risk Engine
      ↓
Risk Score (0–100)
      ↓
ALLOW / VERIFY / BLOCK
      ↓
Prevention & Verification
```

## Technology Stack

### Frontend

* React
* TypeScript
* Vite
* Tailwind CSS
* Recharts

### Backend

* Python
* FastAPI
* WebSockets

### AI / Audio

* PyTorch
* AASIST
* Resemblyzer
* Librosa
* Torchaudio

### Storage

* SQLite for MVP

### Development

* Git
* GitHub

## Project Structure

```text
voice-impersonation-detector/
│
├── backend/       # FastAPI backend
├── frontend/      # React frontend
├── ml/            # ML models and inference
│   ├── aasist/
│   ├── inference/
│   └── speaker/
├── data/          # Local datasets (not committed)
├── docs/          # Architecture and documentation
├── tests/         # Tests
├── TASKS.md       # Team task tracking
├── README.md
└── .gitignore
```

## Getting Started

Setup instructions will be added as the individual components are implemented.

## Architecture

See [`docs/architecture.md`](docs/architecture.md).

## Important

Model checkpoints, datasets, audio recordings, generated files, and other large/local files should not be committed to Git.

## Team

SIH 2026 Team
