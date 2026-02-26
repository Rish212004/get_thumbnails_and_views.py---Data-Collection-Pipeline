import sys
# --- 🛑 NUCLEAR FIX ---
# Forces the script to ignore your broken TensorFlow installation
sys.modules['tensorflow'] = None 

import os
import csv
import re
import glob
import warnings
import tkinter as tk
from tkinter import filedialog
import yt_dlp
import torch
from transformers import pipeline

# --- CONFIGURATION ---
IMAGE_FORMAT = 'jpg'
WHISPER_MODEL = "distil-whisper/distil-small.en"
BATCH_SIZE = 8

# Silence warnings
warnings.filterwarnings("ignore")

def select_csv_file():
    root = tk.Tk()
    root.withdraw()
    print("📂 Waiting for file selection...")
    file_path = filedialog.askopenfilename(
        title="Select your CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return file_path

def load_whisper_model():
    """Loads the AI Transcriber."""
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
        print("   ✅ Fallback AI: Mac GPU (MPS) Enabled")
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        print("   ✅ Fallback AI: NVIDIA GPU Enabled")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("   ⚠️ Fallback AI: Using CPU (Slow)")

    print(f"   ⚡ Loading Whisper Model ('{WHISPER_MODEL}')...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        torch_dtype=torch_dtype,
        device=device
    )
    return pipe

def sanitize_filename(name):
    clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return clean_name.strip()[:150]

def clean_vtt_text(vtt_content):
    lines = vtt_content.splitlines()
    text_lines = []
    seen = set()
    for line in lines:
        if "-->" in line or line.strip() == "" or line.startswith("WEBVTT"): continue
        clean_line = re.sub(r'<[^>]+>', '', line).strip()
        if clean_line and clean_line not in seen:
            text_lines.append(clean_line)
            seen.add(clean_line)
    return " ".join(text_lines)

def transcribe_audio_fallback(pipe, youtube_url, temp_audio_path):
    print("   ⚠️ Subtitle download failed. Switching to Plan B (AI Transcription)...")
    
    # Ensure filename ends in mp3
    if not temp_audio_path.endswith(".mp3"):
        temp_audio_path += ".mp3"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_audio_path.replace(".mp3", ""), # yt-dlp adds extension auto
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}],
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            
        if os.path.exists(temp_audio_path):
            print("   🎧 Transcribing audio...")
            result = pipe(
                temp_audio_path, 
                chunk_length_s=30,
                batch_size=BATCH_SIZE, 
                return_timestamps=True
            )
            # Cleanup
            os.remove(temp_audio_path)
            return result["text"]
        else:
            return "[AUDIO_DOWNLOAD_FAILED]"
            
    except Exception as e:
        print(f"   ❌ Transcription Failed: {e}")
        return "[TRANSCRIPTION_FAILED]"

def process_video(youtube_url, folder, csv_writer, ai_pipe):
    print(f"🔎 Checking: {youtube_url}")
    
    # 1. Fetch Metadata (Robust)
    ydl_opts_meta = {'quiet': True, 'skip_download': True, 'ignoreerrors': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if not info:
                print("   ❌ Video unavailable/Private.")
                return
            title = info.get('title', 'Unknown')
            views = info.get('view_count', 0)
    except Exception as e:
        print(f"   ❌ Metadata Error: {e}")
        return

    clean_title = sanitize_filename(title)
    output_base = os.path.join(folder, clean_title)
    
    print(f"   📝 Title: {title[:40]}...")

    # 2. Download Thumbnail (Keep Separate)
    try:
        ydl_opts_thumb = {
            'skip_download': True,
            'writethumbnail': True,
            'outtmpl': output_base,
            'postprocessors': [{'key': 'FFmpegThumbnailsConvertor', 'format': IMAGE_FORMAT}],
            'quiet': True, 'no_warnings': True
        }
        with yt_dlp.YoutubeDL(ydl_opts_thumb) as ydl:
            ydl.download([youtube_url])
    except:
        print("   ⚠️ Thumbnail download failed.")

    # 3. Attempt Subtitles (Try/Except Block is Critical Here)
    script_text = ""
    got_captions = False

    try:
        ydl_opts_subs = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'outtmpl': output_base,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'vtt',
            'quiet': True, 'no_warnings': True
        }
        with yt_dlp.YoutubeDL(ydl_opts_subs) as ydl:
            ydl.download([youtube_url])
        
        # Check if file exists
        vtt_files = glob.glob(f"{glob.escape(output_base)}*.vtt")
        if vtt_files:
            with open(vtt_files[0], 'r', encoding='utf-8') as f:
                script_text = clean_vtt_text(f.read())
            os.remove(vtt_files[0])
            got_captions = True
            print("   📜 Source: Auto-Captions")

    except Exception:
        # If yt-dlp crashes here, we strictly ignore it and move to fallback
        pass

    # 4. Fallback if Captions Failed
    if not got_captions:
        temp_audio_name = os.path.join(folder, f"temp_{clean_title}")
        script_text = transcribe_audio_fallback(ai_pipe, youtube_url, temp_audio_name)

    # 5. Save Data
    image_filename = f"{clean_title}.{IMAGE_FORMAT}"
    csv_writer.writerow([title, views, script_text, image_filename, youtube_url])
    print("   ✅ Data Saved.\n")

if __name__ == "__main__":
    input_csv = select_csv_file()
    if not input_csv: exit()

    print("\n🚀 Initializing AI System...")
    ai_pipe = load_whisper_model()

    csv_dir = os.path.dirname(input_csv)
    output_folder = os.path.join(csv_dir, "Dataset_Thumbnails")
    output_csv = os.path.join(csv_dir, "complete_dataset.csv")
    
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        writer.writerow(["Title", "Views", "Script_Text", "Thumbnail_File", "Link"])
        
        for row in reader:
            if not row: continue
            url = None
            for item in row:
                if item.strip().startswith("http"):
                    url = item.strip()
                    break
            if url:
                process_video(url, output_folder, writer, ai_pipe)

    print(f"✨ Done! Dataset: {output_csv}")