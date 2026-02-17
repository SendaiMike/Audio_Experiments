#!/usr/bin/env python3
"""
Check for inter-sample peaks (overshoots) after WAV to MP3 conversion.
These peaks can occur during DAC reconstruction even if the original samples don't clip.
"""

import numpy as np
import subprocess
import sys
import os
from pathlib import Path

def get_audio_info(file_path):
    """Get audio file info using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels,codec_name',
        '-of', 'csv=p=0',
        file_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        codec, sample_rate, channels = result.stdout.strip().split(',')
        return {
            'codec': codec,
            'sample_rate': int(sample_rate),
            'channels': int(channels)
        }
    except subprocess.CalledProcessError as e:
        print(f"Error getting file info: {e}")
        return None

def load_audio_as_float(file_path, oversampling=4):
    """
    Load audio file and oversample to detect inter-sample peaks.
    Returns numpy array normalized to -1.0 to 1.0 range.
    """
    # Use ffmpeg to decode and output as 32-bit float PCM
    cmd = [
        'ffmpeg',
        '-i', file_path,
        '-f', 'f32le',  # 32-bit float little-endian
        '-acodec', 'pcm_f32le',
        '-ar', str(192000 * oversampling),  # Oversample to catch ISPs
        '-ac', '2',  # Stereo
        'pipe:1'
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        audio_data = np.frombuffer(result.stdout, dtype=np.float32)
        
        # Reshape to stereo (2 channels)
        if len(audio_data) > 0:
            audio_data = audio_data.reshape(-1, 2)
        
        return audio_data
    
    except subprocess.CalledProcessError as e:
        print(f"Error loading audio: {e}")
        return None

def check_peaks(audio_data, threshold_db=-0.1):
    """
    Check for peaks above threshold.
    
    Args:
        audio_data: numpy array of audio samples (normalized to -1.0 to 1.0)
        threshold_db: threshold in dBFS (default -0.1 dBFS catches most overshoots)
    
    Returns:
        dict with peak information
    """
    if audio_data is None or len(audio_data) == 0:
        return None
    
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold_db / 20)
    
    # Get absolute values
    abs_audio = np.abs(audio_data)
    
    # Find peaks for each channel
    results = {
        'threshold_db': threshold_db,
        'threshold_linear': threshold_linear,
        'channels': []
    }
    
    for ch in range(audio_data.shape[1]):
        channel_data = abs_audio[:, ch]
        max_peak = np.max(channel_data)
        max_peak_db = 20 * np.log10(max_peak) if max_peak > 0 else -np.inf
        
        # Find samples above threshold
        peaks_above = channel_data > threshold_linear
        num_peaks = np.sum(peaks_above)
        
        # Get peak locations
        peak_indices = np.where(peaks_above)[0]
        
        channel_result = {
            'channel': ch + 1,
            'max_peak_linear': float(max_peak),
            'max_peak_db': float(max_peak_db),
            'num_samples_over': int(num_peaks),
            'clipping': max_peak >= 1.0,
            'overshoot_detected': num_peaks > 0
        }
        
        if num_peaks > 0 and len(peak_indices) > 0:
            channel_result['first_peak_sample'] = int(peak_indices[0])
            channel_result['last_peak_sample'] = int(peak_indices[-1])
        
        results['channels'].append(channel_result)
    
    # Overall status
    results['has_overshoots'] = any(ch['overshoot_detected'] for ch in results['channels'])
    results['has_clipping'] = any(ch['clipping'] for ch in results['channels'])
    
    return results

def analyze_file(wav_path, mp3_bitrate=320, threshold_db=-0.1, keep_mp3=False):
    """
    Analyze a WAV file for potential MP3 conversion overshoots.
    
    Args:
        wav_path: path to input WAV file
        mp3_bitrate: bitrate for MP3 conversion (default 320kbps)
        threshold_db: detection threshold in dBFS
        keep_mp3: keep the temporary MP3 file after analysis
    """
    wav_path = Path(wav_path)
    
    if not wav_path.exists():
        print(f"Error: File not found: {wav_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {wav_path.name}")
    print(f"{'='*60}\n")
    
    # Get original file info
    info = get_audio_info(str(wav_path))
    if info:
        print(f"Original format: {info['codec']}, {info['sample_rate']}Hz, {info['channels']} channels")
    
    # Check original WAV
    print(f"\nChecking original WAV file...")
    wav_data = load_audio_as_float(str(wav_path))
    wav_results = check_peaks(wav_data, threshold_db)
    
    if wav_results:
        for ch in wav_results['channels']:
            print(f"  Channel {ch['channel']}: Peak = {ch['max_peak_db']:.2f} dBFS", end="")
            if ch['clipping']:
                print(" [CLIPPING]")
            elif ch['overshoot_detected']:
                print(f" [OVERSHOOT: {ch['num_samples_over']} samples]")
            else:
                print(" [OK]")
    
    # Convert to MP3
    mp3_path = wav_path.with_suffix('.mp3')
    if mp3_path.exists() and not keep_mp3:
        mp3_path = wav_path.parent / f"{wav_path.stem}_temp.mp3"
    
    print(f"\nConverting to MP3 ({mp3_bitrate}kbps)...")
    
    convert_cmd = [
        'ffmpeg',
        '-i', str(wav_path),
        '-codec:a', 'libmp3lame',
        '-b:a', f'{mp3_bitrate}k',
        '-y',  # Overwrite
        str(mp3_path)
    ]
    
    try:
        subprocess.run(convert_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting to MP3: {e}")
        return None
    
    # Check MP3
    print(f"Checking MP3 file for inter-sample peaks...")
    mp3_data = load_audio_as_float(str(mp3_path), oversampling=4)
    mp3_results = check_peaks(mp3_data, threshold_db)
    
    if mp3_results:
        for ch in mp3_results['channels']:
            print(f"  Channel {ch['channel']}: Peak = {ch['max_peak_db']:.2f} dBFS", end="")
            if ch['clipping']:
                print(" [CLIPPING]")
            elif ch['overshoot_detected']:
                print(f" [OVERSHOOT: {ch['num_samples_over']} samples]")
            else:
                print(" [OK]")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if mp3_results:
        if mp3_results['has_clipping']:
            print("⚠️  CLIPPING DETECTED - Audio exceeds 0 dBFS")
            print("   Recommendation: Reduce gain before MP3 conversion")
        elif mp3_results['has_overshoots']:
            print("⚠️  INTER-SAMPLE PEAKS DETECTED")
            print("   These peaks may cause distortion on some playback systems")
            print("   Recommendation: Apply true-peak limiting before MP3 conversion")
        else:
            print("✓  No overshoots detected - Safe for streaming")
    
    print(f"\nThreshold used: {threshold_db} dBFS")
    
    # Cleanup temp MP3 if requested
    if not keep_mp3 and mp3_path.name.endswith('_temp.mp3'):
        mp3_path.unlink()
        print(f"\nTemporary MP3 removed: {mp3_path.name}")
    else:
        print(f"\nMP3 saved: {mp3_path}")
    
    return {
        'wav_results': wav_results,
        'mp3_results': mp3_results,
        'mp3_path': mp3_path
    }

def main():
    # Check if filepath provided as argument
    if len(sys.argv) >= 2:
        wav_path = sys.argv[1]
        bitrate = int(sys.argv[2]) if len(sys.argv) > 2 else 320
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else -0.1
    else:
        # Interactive mode - prompt for filepath
        print("MP3 Inter-Sample Peak Checker")
        print("=" * 60)
        wav_path = input("\nEnter path to WAV file: ").strip()
        
        # Remove quotes if user pastes path with quotes
        wav_path = wav_path.strip('"\'')
        
        # Optional: ask for custom settings
        bitrate_input = input("MP3 bitrate in kbps (default 320): ").strip()
        bitrate = int(bitrate_input) if bitrate_input else 320
        
        threshold_input = input("Detection threshold in dBFS (default -0.1): ").strip()
        threshold = float(threshold_input) if threshold_input else -0.1
    
    # Validate file exists
    if not os.path.exists(wav_path):
        print(f"\nError: File not found: {wav_path}")
        sys.exit(1)
    
    analyze_file(wav_path, mp3_bitrate=bitrate, threshold_db=threshold, keep_mp3=False)

if __name__ == "__main__":
    main()
