Audio Experiments
Programmatic audio processing utilities built to solve real problems I encountered operating a vocal recording studio.
Background

After 8 years of making audio quality decisions one file at a time, I started building tools to automate workflows while maintaining the perceptual and technical standards I apply manually. These utilities emerged from actual production needs


1. Batch MP3 Converter
Converts audio files to MP3 format with consistent encoding settings across large batches.

Why I built this: Studio deliverables often require MP3 format, but manual conversion through DAWs is time-consuming and inconsistent. This ensures every file uses the same encoder settings while processing dozens of files in parallel.

Features:

Batch processing with parallel execution
Consistent bitrate and sample rate settings
Preserves metadata
Quality validation after conversion


2. Batch ML Preprocessing Pipeline

Prepares audio files for machine learning applications with consistent formatting and quality standards.

Why I built this: Working on AI voice data collection projects exposed the gap between studio-quality recordings and ML-ready datasets. Manual preprocessing doesn't scale to hundreds of files.
Features:

Standardized sample rate conversion (typically 16kHz)
Loudness normalization
Silence trimming from start/end
Format standardization (mono/stereo, bit depth)
Quality validation and reporting

Normalize loudness across all files
Convert to consistent sample rate
Trim silence
Convert to mono if needed
Generate quality report


3. ISP Overshoot Detector
Identifies intersample peak (ISP) overshoot in MP3 conversions. the hidden clipping that occurs between samples during lossy encoding. A common issue for music clients after uploading their approved master to DSPs only to be disapointed by distortion occuring when streaming.

Why I built this: MP3 encoding can introduce clipping that doesn't show up in the original WAV file's peak meters. This matters for broadcast and streaming where even brief overshoot can cause audible distortion. I needed a way to catch this before delivery.
What it detects:
Intersample peaks occur when the continuous analog signal reconstructed from digital samples exceeds 0dBFS, even though no individual sample does. MP3 encoding can create these peaks through its reconstruction filters.
Features:

Scans MP3 files for true peak levels (not just sample peaks)
Reports overshoot amount in dB
Batch processing across directories
Generates reports showing which files need re-encoding with more headroom


Technical context:
Most DAWs show sample peak meters, not true peak. A file can show -0.1 dBFS on your meters but actually clip when converted to MP3. This tool catches that.
Requirements
bashffmpeg  # Install via: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)
python 3.8+
Python dependencies:
bashpip install pydub numpy scipy
Why These Tools?
The Scaling Problem
In a studio environment, I could manually check every file, adjust levels, and ensure quality. But working with larger  datasets or batch client deliverables made manual processing impractical.
The Quality Problem
Automation without understanding degrades quality. These tools encode the audio engineering decisions I'd make manually:

What sample rate serves the use case?
What loudness target maintains clarity without crushing dynamics?
How much headroom prevents encoding artifacts?


I'm actively expanding these tools to handle:

More sophisticated artifact detection (frequency imbalances, phase issues)
ACX audiobook compliance validation
Automated quality scoring systems

The goal: Convert subjective audio engineering expertise into programmatic systems that can operate at scale while maintaining the standards I've built over 8 years of professional audio work.
