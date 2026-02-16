#!/usr/bin/env python3
"""
Audio preprocessing pipeline for speech recognition ML training.
Handles batch processing with standard ML-ready transforms.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import argparse
from tqdm import tqdm
import json

class SpeechMLPreprocessor:
    """
    Preprocesses audio files for speech recognition model training.
    """
    
    def __init__(
        self,
        target_sr: int = 16000,           # sample rate
        mono: bool = True,                # Mono Audio
        normalize: bool = True,           # Peak normalization
        trim_silence: bool = True,        # Remove leading/trailing silence
        vad_threshold: float = 0.025,     # Voice activity detection threshold
        min_duration: float = 0.5,        # Minimum clip length in seconds
        max_duration: float = 360.0,      # Maximum clip length
        output_format: str = 'wav',       # Output format (wav, flac)
        bit_depth: str = 'PCM_16'         # Bit depth for output
    ):
        self.target_sr = target_sr
        self.mono = mono
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.vad_threshold = vad_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.output_format = output_format
        self.bit_depth = bit_depth
        
    def process_file(
        self, 
        input_path: str, 
        output_path: str
    ) -> Tuple[bool, Optional[dict]]:
        """
        Process a single audio file through the ML preprocessing pipeline.
        
        Returns:
            (success, metadata) tuple
        """
        try:
            # 1. LOAD AUDIO
            # Load at native sample rate first, then resample (higher quality)
            audio, sr = librosa.load(input_path, sr=None, mono=self.mono)
            
            # 2. RESAMPLE to target sample rate
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # 3. CONVERT TO MONO (if stereo and mono=True)
            if audio.ndim > 1 and self.mono:
                audio = librosa.to_mono(audio)
            
            # 4. TRIM SILENCE (leading/trailing)
            if self.trim_silence:
                audio, trim_indices = librosa.effects.trim(
                    audio, 
                    top_db=40,  # Aggressive silence threshold
                    frame_length=2048,
                    hop_length=512
                )
            
            # 5. DURATION FILTERING
            duration = len(audio) / sr
            if duration < self.min_duration or duration > self.max_duration:
                return False, {
                    'error': 'duration_out_of_range',
                    'duration': duration,
                    'min': self.min_duration,
                    'max': self.max_duration
                }
            
            # 6. VOICE ACTIVITY DETECTION (simple energy-based)
            # TO DO: IMLPLEMNT VAD MODEL in future
            rms = librosa.feature.rms(y=audio)[0]
            speech_ratio = np.sum(rms > self.vad_threshold) / len(rms)
            
            if speech_ratio < 0.3:  # Less than 30% speech content
                return False, {
                    'error': 'insufficient_speech',
                    'speech_ratio': float(speech_ratio)
                }
            
            # 7. NORMALIZE AUDIO
            if self.normalize:
                # Peak normalization to -1.0 dB to prevent clipping
                peak = np.abs(audio).max()
                if peak > 0:
                    audio = audio / peak * 0.95
            
         
            
            # 8. SAVE PROCESSED AUDIO
            sf.write(
                output_path,
                audio,
                sr,
                subtype=self.bit_depth,
                format=self.output_format.upper()
            )
            
            # 10. GENERATE METADATA
            metadata = {
                'original_path': str(input_path),
                'output_path': str(output_path),
                'sample_rate': sr,
                'duration': float(duration),
                'channels': 1 if self.mono else audio.shape[0] if audio.ndim > 1 else 1,
                'speech_ratio': float(speech_ratio),
                'peak_amplitude': float(np.abs(audio).max()),
                'rms_level': float(np.sqrt(np.mean(audio**2))),
                'success': True
            }
            
            return True, metadata
            
        except Exception as e:
            return False, {
                'error': str(e),
                'original_path': str(input_path)
            }

def batch_process(
    input_dir: str,
    output_dir: str,
    config: Optional[dict] = None
):
    """
    Batch process all audio files in input directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = SpeechMLPreprocessor(**(config or {}))
    
    # Find all audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    audio_files = [
        f for f in input_path.rglob('*') 
        if f.suffix.lower() in audio_extensions
    ]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process files
    results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    for audio_file in tqdm(audio_files, desc="Processing audio"):
        # Maintain directory structure
        relative_path = audio_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix(f'.{preprocessor.output_format}')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        success, metadata = preprocessor.process_file(str(audio_file), str(output_file))
        
        if success:
            results['successful'].append(metadata)
        elif 'error' in metadata and metadata['error'] in ['duration_out_of_range', 'insufficient_speech']:
            results['skipped'].append(metadata)
        else:
            results['failed'].append(metadata)
    
    # Save processing report
    report_path = output_path / 'preprocessing_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'total': len(audio_files),
                'successful': len(results['successful']),
                'failed': len(results['failed']),
                'skipped': len(results['skipped'])
            },
            'config': config or {},
            'details': results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  ✓ Successful: {len(results['successful'])}")
    print(f"  ⊘ Skipped: {len(results['skipped'])}")
    print(f"  ✗ Failed: {len(results['failed'])}")
    print(f"  Report saved to: {report_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess audio for speech ML training')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate (default: 16000)')
    parser.add_argument('--no-normalize', action='store_true', help='Skip normalization')
    parser.add_argument('--no-trim', action='store_true', help='Skip silence trimming')
    parser.add_argument('--min-duration', type=float, default=0.5, help='Minimum duration in seconds')
    parser.add_argument('--max-duration', type=float, default=360.0, help='Maximum duration in seconds')
    parser.add_argument('--format', type=str, default='wav', choices=['wav', 'flac'], help='Output format')
    
    args = parser.parse_args()
    
    # PROMPT FOR INPUT FOLDER
    print("\n" + "="*60)
    print("AUDIO ML PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    input_dir = input("Enter path to input folder containing audio files: ").strip()
    
    # Validate input directory exists
    if not os.path.exists(input_dir):
        print(f"\n❌ Error: Input directory '{input_dir}' does not exist.")
        exit(1)
    
    # CREATE OUTPUT FOLDER
    # Auto-generate output folder name based on input folder
    input_path = Path(input_dir)
    suggested_output = input_path.parent / f"{input_path.name}_processed"
    
    output_prompt = input(f"\nOutput folder (press Enter for '{suggested_output}'): ").strip()
    output_dir = output_prompt if output_prompt else str(suggested_output)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"\nProcessing settings:")
    print(f"  Sample rate: {args.sr} Hz")
    print(f"  Normalize: {not args.no_normalize}")
    print(f"  Trim silence: {not args.no_trim}")
    print(f"  Duration range: {args.min_duration}s - {args.max_duration}s")
    print(f"  Output format: {args.format}")
    
    proceed = input("\nProceed with processing? [Y/n]: ").strip().lower()
    if proceed and proceed != 'y':
        print("\nCancelled.")
        exit(0)
    
    print()  # Blank line before progress bar
    
    config = {
        'target_sr': args.sr,
        'normalize': not args.no_normalize,
        'trim_silence': not args.no_trim,
        'min_duration': args.min_duration,
        'max_duration': args.max_duration,
        'output_format': args.format
    }
    
    batch_process(input_dir, output_dir, config)
   