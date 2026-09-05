# System Architecture

## Overview

Our system detects AI-generated/voice-cloned speech during voice interactions and calculates a real-time risk score to prevent impersonation attacks.

## Architecture

```text
Voice / Audio
      ↓
Audio Preprocessing
      ↓
 ┌───────────────┬──────────────────┐
 ↓               ↓                  ↓
AASIST       Speaker            Context
Deepfake     Verification       Analysis
Detector       Module             |
 ↓               ↓                ↓
Synthetic     Speaker Match   Context Risk
Score            Score             |
 └───────────────┴──────────────────┘
                 ↓
            Risk Engine
                 ↓
        Risk Score (0–100)
                 ↓
       ┌─────────┼─────────┐
       ↓         ↓         ↓
     ALLOW     VERIFY     BLOCK
                 ↓         ↓
            Verification  Alert +
             Workflow    Prevention
```

## Main Components

### 1. Audio Processing

* Accepts recorded or streamed audio.
* Converts audio to the required format.
* Resamples and normalizes audio.
* Splits audio into segments for near-real-time analysis.

### 2. AI Voice Detection

* Uses a pretrained AASIST model.
* Determines whether speech is likely genuine or AI-generated.
* Produces a `synthetic_score` between 0 and 1.

### 3. Speaker Verification

* Compares the incoming speaker with an enrolled legitimate speaker.
* Uses voice embeddings and similarity comparison.
* Produces a `speaker_match_score`.

### 4. Context Analysis

Detects risk indicators such as:

* Transfer requests
* OTP requests
* Urgent/immediate requests
* Unusual activity

### 5. Risk Engine

Combines:

* Synthetic voice score
* Speaker mismatch
* Context risk

and produces a score from 0–100.

Initial thresholds:

```text
Risk < 30       → ALLOW
Risk 30–70      → VERIFY
Risk > 70       → BLOCK
```

### 6. Prevention Layer

```text
LOW RISK
   ↓
Allow action

MEDIUM RISK
   ↓
Request secondary verification

HIGH RISK
   ↓
Pause sensitive action
   ↓
Generate alert
   ↓
Require verification
   ↓
Allow / Reject
```

### 7. Backend

FastAPI provides:

```text
POST /enroll
POST /analyze-call
GET  /alerts
WebSocket /ws/call
```

### 8. Frontend

The React dashboard displays:

* Live risk score
* Synthetic voice score
* Speaker match score
* Risk status
* Alerts
* Verification workflow
* Decision history

## Technology Stack

* React + TypeScript
* FastAPI
* Python
* PyTorch
* AASIST
* Resemblyzer
* Librosa
* Torchaudio
* WebSockets
* SQLite
* Git + GitHub
