import os
import subprocess

def convert_wavs_to_mp3(folder_path):
    """Convert all WAV files to MP3 and save in 'mp3s' subfolder"""
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Create 'mp3s' output folder
    output_folder = os.path.join(folder_path, 'mp3s')
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}\n")
    
    # Find all WAV files
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    
    if len(wav_files) == 0:
        print(f"No WAV files found in '{folder_path}'")
        return
    
    print(f"Found {len(wav_files)} WAV files")
    print("Converting to MP3...\n")
    
    # Convert each file
    success_count = 0
    for filename in wav_files:
        input_path = os.path.join(folder_path, filename)
        output_filename = filename.replace('.wav', '.mp3').replace('.WAV', '.mp3')
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"Converting: {filename}")
        
        try:
            subprocess.run([
                'ffmpeg', '-i', input_path,
                '-b:a', '320k',
                '-y',
                output_path
            ], check=True, capture_output=True)
            
            print(f"✓ Saved: mp3s/{output_filename}\n")
            success_count += 1
            
        except subprocess.CalledProcessError:
            print(f"✗ Failed to convert {filename}\n")
    
    print(f"Done! Successfully converted {success_count}/{len(wav_files)} files.")
    print(f"MP3 files saved in: {output_folder}")

# Get folder path from user
print("WAV to MP3 Converter")
print("-" * 40)
folder = input("Enter path to folder with WAV files: ")

# Remove quotes if user dragged folder (macOS adds them)
folder = folder.strip().strip("'").strip('"')

convert_wavs_to_mp3(folder)