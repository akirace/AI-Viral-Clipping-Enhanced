import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from services.video_processor import VideoProcessor
from styles.caption_styles import CAPTION_STYLES
from config import OUTPUT_DIR, TEMP_DIR

# Ensure output is not buffered
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Function to print with immediate flush
def log(message):
    """Prints a message to the console with immediate flushing."""
    print(message, flush=True)

def display_caption_styles():
    """Displays the available caption styles to the user."""
    styles = CAPTION_STYLES
    log("\n🎨 Available Caption Styles (No Strokes):")
    log("=" * 50)
    for i, (key, value) in enumerate(styles.items(), 1):
        log(f"{i}. {key} - {value['name']}")
    log("=" * 50)
    return list(styles.keys())

def list_temp_videos():
    """Lists available video files in the temp directory."""
    valid_extensions = {'.mp4', '.mkv', '.mov', '.avi', '.webm'}
    files = [f for f in TEMP_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

def validate_youtube_url(url):
    """
    Validates that the provided URL is a valid YouTube URL.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is a valid YouTube URL, False otherwise.
    """
    if not url:
        return False
        
    # Basic validation
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
        
    # Check if it's a YouTube domain
    if not ('youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc):
        return False
        
    return True

def main(url=None):
    """
    The main function of the YouTube Viral Clipper application.

    Args:
        url (str, optional): The YouTube URL to process. Defaults to None.
    """
    log("🚀 === YOUTUBE VIRAL CLIPPER === 🚀")
    log("✨ Features: AI Clip Selection | Smart Face Tracking | Word-by-Word Captions")
    log("")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    log(f"📁 Output directory: {OUTPUT_DIR}")
    log(f"📁 Temp directory: {TEMP_DIR}")

    if not url:
        print("\nChoose input source:")
        print("1. Download from YouTube URL")
        print("2. Select local video from /temp")
        choice = input("Enter choice (1 or 2) [1]: ").strip() or "1"

        if choice == "2":
            videos = list_temp_videos()
            if not videos:
                log("❌ No videos found in temp directory.")
                return
            
            print("\nAvailable local videos:")
            for i, vid in enumerate(videos, 1):
                size_mb = vid.stat().st_size / (1024 * 1024)
                print(f"{i}. {vid.name} ({size_mb:.2f} MB)")
            
            while True:
                try:
                    vid_choice = input("\nSelect video number: ").strip()
                    idx = int(vid_choice) - 1
                    if 0 <= idx < len(videos):
                        url = str(videos[idx].absolute())
                        log(f"✅ Selected local video: {videos[idx].name}")
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            url = input("🔗 Enter YouTube URL: ").strip()
            if not url:
                log("❌ Error: No URL provided.")
                return
            # Validate URL
            if not validate_youtube_url(url):
                log("❌ Error: Invalid YouTube URL. Please provide a valid YouTube URL.")
                return

    try:
        clips_input = input("📊 Number of viral clips [3] (enter '0', 'all' or 'auto' for unlimited): ").strip().lower() or "3"
        if clips_input in ['0', 'all', 'auto']:
            num_clips = 0
            log("♾️  Mode: Unlimited/Auto-Select Clips")
        else:
            num_clips = int(clips_input)
        
        max_duration = int(input("⏱️  Max seconds per clip [60]: ") or "60")
        min_duration = int(input("⏱️  Min seconds per clip [20]: ") or "20")

        if min_duration >= max_duration:
            log("❌ Error: Min duration must be less than max duration.")
            return

        style_keys = display_caption_styles()
        style_choice = input(f"🎨 Select caption style (1-{len(style_keys)}) [1]: ").strip() or "1"

        try:
            style_index = int(style_choice) - 1
            if 0 <= style_index < len(style_keys):
                selected_style = style_keys[style_index]
            else:
                selected_style = 'clean_white'
                log("⚠️  Invalid choice, using default: Clean White")
        except:
            selected_style = 'clean_white'
            log("⚠️  Invalid input, using default: Clean White")

        log("\n" + "="*70)
        log("🎬 STARTING VIDEO PROCESSING...")
        log("="*70)

        log("⏳ Initializing video processor...")
        processor = VideoProcessor(caption_style=selected_style)
        log("✅ Video processor initialized")
        
        # Add a small delay to ensure logs are displayed
        time.sleep(0.5)
        
        outputs, title = processor.process_video(url, num_clips, min_duration, max_duration)

        log("\n" + "="*70)
        log("🎉 VIRAL CLIPS GENERATED!")
        log("="*70)
        log(f"📹 Source: '{title}'")
        log(f"📁 Location: {OUTPUT_DIR.resolve()}")
        log(f"🎨 Caption Style: {processor.caption_maker.styles[selected_style]['name']}")

        total_size = 0
        log("\n📋 Generated Clips:")
        for f in outputs:
            path = Path(f)
            try:
                if path.exists():
                    size_bytes = path.stat().st_size
                    size_mb = size_bytes / (1024 * 1024)
                    total_size += size_mb
                    log(f"  - {path.name} ({size_mb:.2f} MB)")
            except Exception as e:
                log(f"  - {path.name} (size unknown: {e})")
        log(f"\n💾 Total Size: {total_size:.2f} MB")
        log("\n✅ Done! Enjoy your viral clips!")

    except ValueError as e:
        log(f"\n❌ Input Error: {e}")
    except FileNotFoundError as e:
        log(f"\n❌ File Error: {e}")
    except Exception as e:
        log(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        log("\nPlease check the error message above and try again.")

if __name__ == "__main__":
    try:
        # Check if URL was provided as command-line argument
        url = None
        if len(sys.argv) > 1:
            url = sys.argv[1].strip()
            # Basic URL validation
            if not url.startswith('http'):
                url = 'https://' + url
        
        # Create necessary directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(Path("./temp"), exist_ok=True)
        
        main(url)
    except KeyboardInterrupt:
        print("\n\n🛑 Process interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"\n\n❌ Critical error: {e}")
        print("Please report this issue if it persists.")
