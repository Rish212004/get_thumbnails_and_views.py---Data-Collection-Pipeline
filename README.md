Purpose: Creates a complete dataset by collecting thumbnails, view counts, and transcripts

Key Features:

Smart Caption Retrieval: Attempts to extract auto-generated captions from YouTube
VTT Parsing: Cleans up subtitle files for readable text
Fallback Transcription: Uses Whisper AI if captions aren't available
Thumbnail Download: Saves video thumbnails as JPG files
Robust Error Handling: Gracefully handles private/unavailable videos
Dataset Creation: Outputs structured CSV with all collected data
Workflow:

User selects input CSV with YouTube URLs
For each video:
Fetches metadata (title, view count)
Downloads thumbnail image
Attempts to get captions (VTT format)
Falls back to AI transcription if needed
Writes all data to output CSV
Creates organized output folder with thumbnails and metadata CSV
