from faster_whisper import WhisperModel
import torch
import os
from config import WHISPER_MODEL

class WhisperSingleton:
    """
    A singleton class for transcribing audio using the faster-whisper library.

    This class ensures that the Whisper model is loaded only once and provides
    a method to transcribe audio from a video file.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """Loads the faster-whisper model with optimized settings."""
        if self._model is None:
            print(f"Loading faster-whisper model ({WHISPER_MODEL})... (one time only)")
            
            # Strict CUDA check
            if not torch.cuda.is_available():
                print("\n❌ CRITICAL ERROR: CUDA (GPU) is not available!")
                print("   The user requested FORCED CUDA usage.")
                print("   Please ensure you have installed:")
                print("   1. NVIDIA Drivers")
                print("   2. PyTorch with CUDA support (pip install torch --index-url https://download.pytorch.org/whl/cu118)")
                raise RuntimeError("CUDA is required but not available. Aborting.")

            try:
                # Set environment variable to reduce memory usage
                os.environ["OMP_NUM_THREADS"] = "1"
                
                # Check for Flash Attention support (RTX 5050 usually supports FP16)
                compute_type = "float16"
                device = "cuda"
                
                print(f"🚀 FORCING CUDA USAGE on {torch.cuda.get_device_name(0)}")
                
                # Load the model with optimized settings
                self._model = WhisperModel(
                    WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4, # Still need some CPU threads for preprocessing
                    num_workers=2
                )
                print(f"✅ faster-whisper model loaded and cached on: {device} with {compute_type} precision")
            except Exception as e:
                print(f"\n❌ FATAL ERROR loading Whisper model on GPU: {e}")
                print("   Re-run installation or check your drivers.")
                raise e


    def transcribe(self, video_path):
        """
        Transcribes the audio from a video file.

        Args:
            video_path (str): The path to the video file.

        Returns:
            tuple: A tuple containing a list of words with timestamps, the full
                   transcript, and a list of segments.
        """
        print("🎵 Transcribing video...")
        try:
            print("⏳ Initializing transcription with faster-whisper...")
            # faster-whisper has a different API
            print("⏳ Starting audio processing (this may take a while)...")
            segments, info = self._model.transcribe(
                str(video_path), 
                word_timestamps=True,
                vad_filter=True,  # Voice activity detection to skip silence
                vad_parameters={"min_silence_duration_ms": 500},  # Adjust silence detection
                language="en",
                beam_size=1,  # Reduce beam size for faster processing
                best_of=1,    # Only keep the best result
                temperature=0  # Disable sampling for deterministic results
            )
            print("✅ Audio processing complete, now extracting words and segments...")
            
            # Process segments and words
            words = []
            segments_list = []
            full_text = ""
            
            segment_count = 0
            word_count = 0
            
            print("⏳ Processing transcription segments...")
            for segment in segments:
                segment_count += 1
                if segment_count % 10 == 0:
                    print(f"⏳ Processed {segment_count} segments so far...")
                    
                segments_list.append({
                    'id': segment.id,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                })
                full_text += segment.text + " "
                
                # Extract word timestamps
                if hasattr(segment, 'words') and segment.words:
                    for word_info in segment.words:
                        word = word_info.word.strip().upper()
                        if word:
                            words.append({
                                'word': word, 
                                'start': word_info.start, 
                                'end': word_info.end
                            })
                            word_count += 1
            
            print(f"✅ Transcription complete! Found {len(words)} words in {len(segments_list)} segments")
            print(f"📝 Transcript length: {len(full_text)} characters")
            return words, full_text, segments_list

        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return [], "", []
