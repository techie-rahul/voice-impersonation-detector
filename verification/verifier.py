"""Speaker verification module using Resemblyzer.

Provides functions to extract speaker embeddings from audio files and
verify whether two audio samples belong to the same speaker by computing
cosine similarity between their embeddings.
"""

import os

from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine

encoder = VoiceEncoder()


def get_embedding(audio_path):
    """Extract a speaker embedding from an audio file.

    Args:
        audio_path: Path to the audio file (WAV, FLAC, etc.).

    Returns:
        A NumPy array representing the speaker embedding.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the audio file cannot be read or processed.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        wav = preprocess_wav(audio_path)
    except Exception as e:
        raise RuntimeError(f"Could not read or process audio file '{audio_path}': {e}") from e
    return encoder.embed_utterance(wav)


def verify_speaker(audio_path, enrolled_embedding):
    """Verify whether an audio sample matches an enrolled speaker.

    Args:
        audio_path: Path to the audio file to verify.
        enrolled_embedding: NumPy array of the enrolled speaker's embedding
            (obtained from ``get_embedding``).

    Returns:
        A float in [0, 1] representing cosine similarity. Higher values
        indicate the two samples are more likely from the same speaker.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the audio file cannot be read or processed.
    """
    new_embedding = get_embedding(audio_path)
    similarity = 1 - cosine(new_embedding, enrolled_embedding)
    return similarity
