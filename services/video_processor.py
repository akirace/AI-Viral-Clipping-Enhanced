from pathlib import Path
import os
from moviepy.editor import VideoFileClip

from config import OUTPUT_DIR, TEMP_DIR, GEMINI_API_KEY
from services.youtube_downloader import YouTubeDownloader
from services.whisper_transcriber import WhisperSingleton
from services.gemini_selector import GeminiSelector
from services.face_tracker import FaceTracker
from services.caption_maker import CaptionMaker
from utils.helpers import generate_random_clips, cleanup_temp_files


class VideoProcessor:
    """
    Orchestrates the entire video processing pipeline.

    This class initializes and manages all the services required for
    downloading, transcribing, selecting clips, tracking faces, and
    adding captions to a YouTube video.
    """
    def __init__(self, caption_style='clean_white'):
        """
        Initializes the VideoProcessor with a specific caption style.

        Args:
            caption_style (str, optional): The style of captions to use.
                                            Defaults to 'clean_white'.
        """
        self.downloader = YouTubeDownloader()
        self.transcriber = WhisperSingleton()
        self.ai_selector = GeminiSelector(api_key=GEMINI_API_KEY)
        self.face_tracker = FaceTracker()
        self.caption_maker = CaptionMaker(caption_style)

    def process_video(self, url, num_clips, min_duration, max_duration):
        """
        Processes a video (URL or local path) to generate viral clips.

        Args:
            url (str): The URL of the YouTube video OR a local file path.
            num_clips (int): The number of clips to generate.
            min_duration (int): The minimum duration of each clip.
            max_duration (int): The maximum duration of each clip.

        Returns:
            tuple: A tuple containing a list of output file paths and the
                   title of the video.
        """
        # Check if url is actually a local file path
        if os.path.exists(url) and os.path.isfile(url):
            print(f"📂 Using local video file: {url}")
            video_path = Path(url)
            title = video_path.stem
            
            # Get duration using moviepy
            try:
                with VideoFileClip(str(video_path)) as clip:
                    duration = clip.duration
                print(f"✅ Local video loaded: {title} ({duration:.1f}s)")
            except Exception as e:
                raise ValueError(f"Could not read local video file: {e}")
        else:
            print("📥 Downloading video...")
            video_path, title, duration = self.downloader.download(url)
            print(f"✅ Download complete: {title} ({duration:.1f}s)")

        print("🎵 Starting transcription with Whisper...")
        words, transcript, segments = self.transcriber.transcribe(video_path)
        print(f"✅ Transcription processing complete")

        if not segments:
            print("❌ Transcription failed. Using random clips.")
            print("🎲 Generating random clips...")
            clip_specs = generate_random_clips(duration, num_clips, min_duration, max_duration)
            print(f"✅ Generated {len(clip_specs)} random clips")
        else:
            print("🧠 Using AI to select the most viral clips...")
            clip_specs = self.ai_selector.select_clips(
                segments, duration, num_clips, min_duration, max_duration
            )
            print(f"✅ AI selected {len(clip_specs)} viral clips")

        if not clip_specs:
            raise ValueError("Could not select any clips from the video.")

        output_files = []
        print(f"\n🎬 Processing {len(clip_specs)} viral clips...")

        for i, clip_info in enumerate(clip_specs, 1):
            start = clip_info['start']
            end = clip_info['end']
            title_text = clip_info.get('title', f'Clip {i}')
            virality_score = clip_info.get('virality_score', 0)

            print(f"\n📹 Clip {i}/{len(clip_specs)}: {title_text}")
            print(f"    ⭐ Virality Score: {virality_score}/100")
            print(f"    ⏱️  Time: {start:.1f}s to {end:.1f}s")
            print(f"    🎨 Caption Style: {self.caption_maker.styles[self.caption_maker.selected_style]['name']}")

            try:
                with VideoFileClip(str(video_path)) as video:
                    clip = video.subclip(start, end)

                    print(f"    🎯 Applying intelligent face tracking...")
                    print(f"    ⏳ Analyzing frames for face detection...")
                    clip = self.face_tracker.track_and_crop(clip)
                    print(f"    ✅ Face tracking and cropping complete")

                    if words:
                        print(f"    📝 Adding word-by-word captions...")
                        clip = self.caption_maker.add_captions(clip, words, start)

                    filename = f"clip_{i}_{virality_score}pts_{Path(video_path).stem}.mp4"
                    output_path = OUTPUT_DIR / filename

                    print(f"    🎥 Encoding with optimized settings...")
                    print(f"    ⏳ Starting video encoding (this may take a while)...")
                    clip.write_videofile(
                        str(output_path),
                        codec='libx264',
                        audio_codec='aac',
                        # Use 'faster' preset for better performance with minimal quality loss
                        preset='faster',
                        # Use slightly higher CRF (20) for better compression with minimal quality loss
                        ffmpeg_params=['-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2'],
                        verbose=True,  # Enable verbose output to show progress
                        logger=None,
                        temp_audiofile=str(TEMP_DIR / f'temp_audio_{i}.m4a'),
                        remove_temp=True,
                        # Use 2 threads for encoding to avoid overloading the system
                        threads=2
                    )
                    print(f"    ✅ Video encoding complete")
                    output_files.append(str(output_path))
                    print(f"    ✅ Saved: {filename}")

            except Exception as e:
                print(f"    ❌ Error processing clip {i}: {e}")
                continue
        
        self.face_tracker.close()
        cleanup_temp_files()

        return output_files, title
