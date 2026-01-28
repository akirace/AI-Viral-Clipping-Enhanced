import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import FACE_DETECTION_MODEL_PATH

class FaceTracker:
    """
    Tracks faces in a video and crops the frame to keep the speaker centered.
    Uses MediaPipe Tasks API (FaceDetector).
    """
    def __init__(self):
        """
        Initializes the FaceTracker with a MediaPipe face detection model.
        Downloads the model if it doesn't exist.
        """
        self.model_path = FACE_DETECTION_MODEL_PATH
        self._ensure_model_exists()

        # Initialize FaceDetector
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        self.detector = vision.FaceDetector.create_from_options(options)

        # Cache for detected faces to avoid reprocessing
        self.face_cache = {}
        print("🎯 Initialized intelligent face tracking with MediaPipe Tasks")

    def _ensure_model_exists(self):
        """Checks if the model file exists, and downloads it if not."""
        if not os.path.exists(self.model_path):
            print(f"    📥 Model file not found at {self.model_path}")
            print("    ⬇️ Downloading Face Detection model from Google...")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("    ✅ Model downloaded successfully.")
            except Exception as e:
                print(f"    ❌ Failed to download model: {e}")
                raise RuntimeError(f"Could not download face detection model: {e}")

    def detect_faces_in_frame(self, frame, frame_time=None):
        """
        Detects faces in a single frame of a video.

        Args:
            frame (numpy.ndarray): The video frame to process (BGR).
            frame_time (float, optional): The timestamp of the frame.

        Returns:
            list: A list of dictionaries, each representing a detected face.
        """
        if frame_time is not None and frame_time in self.face_cache:
            return self.face_cache[frame_time]
            
        try:
            # First pass: Resize frame for faster processing (half size)
            h, w, _ = frame.shape
            scale = 0.5
            small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            detection_result = self.detector.detect(mp_image)

            # Second pass: If no faces found, retry with FULL resolution
            if not detection_result.detections:
                # print("    ⚠️ No faces in scaled frame, retrying full resolution...")
                rgb_frame_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_full)
                detection_result = self.detector.detect(mp_image_full)
                scale = 1.0 # Reset scale so coordinates are correct
            
            # Third pass: Check LEFT half (for podcasts/interviews)
            offset_x = 0
            if not detection_result.detections:
                # print("    ⚠️ No faces in full frame, checking LEFT half...")
                h, w, _ = frame.shape
                left_half = frame[:, :w//2]
                rgb_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
                mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_left)
                detection_result = self.detector.detect(mp_image_left)
                scale = 1.0
                offset_x = 0 # No offset needed for left half

            # Fourth pass: Check RIGHT half
            if not detection_result.detections:
                # print("    ⚠️ No faces in left half, checking RIGHT half...")
                h, w, _ = frame.shape
                right_half = frame[:, w//2:]
                rgb_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB)
                mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_right)
                detection_result = self.detector.detect(mp_image_right)
                scale = 1.0
                offset_x = w // 2 # Add offset for right half detections

            faces = []
            if detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    
                    # Scale back to original size and apply offset
                    x = int(bbox.origin_x / scale) + offset_x
                    y = int(bbox.origin_y / scale)
                    width = int(bbox.width / scale)
                    height = int(bbox.height / scale)

                    center_x = x + width // 2
                    center_y = y + height // 2
                    
                    # Score is a list, usually first one is confidence
                    confidence = detection.categories[0].score if detection.categories else 0.0

                    faces.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height,
                        'confidence': confidence,
                        'area': width * height
                    })

            result = sorted(faces, key=lambda f: f['confidence'] * f['area'], reverse=True)
            
            if frame_time is not None:
                self.face_cache[frame_time] = result
                
            return result
        except Exception as e:
            print(f"    ⚠️ Face detection error: {e}")
            return []

    def smooth_trajectory(self, positions, window_size=5):
        """
        Smoothes a trajectory of positions using a moving average.
        """
        if len(positions) <= window_size:
            return positions

        smoothed = []
        for i in range(len(positions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(positions), i + window_size // 2 + 1)
            window = positions[start_idx:end_idx]

            avg_x = sum(pos[0] for pos in window) / len(window)
            avg_y = sum(pos[1] for pos in window) / len(window)
            smoothed.append((avg_x, avg_y))

        return smoothed

    def track_and_crop(self, clip):
        """
        Tracks faces in a video clip and crops it to keep the speaker centered.
        """
        width, height = clip.size
        target_width = int(height * 9 / 16)
        if target_width % 2 != 0:
            target_width -= 1
        if width <= target_width:
            print("    ⏩ Skipping face tracking - video already in target aspect ratio")
            return clip

        print("    🎯 Analyzing frames for optimal face tracking (MediaPipe Tasks)...")

        self.face_cache = {}
        
        face_positions = []
        # Analyze fewer frames for better performance
        num_samples = min(6, max(3, int(clip.duration / 3)))
        if clip.duration > 10:
            num_samples = min(8, max(6, int(clip.duration / 4)))
        
        print(f"    ⏳ Analyzing {num_samples} frames across {clip.duration:.1f}s of video...")
            
        sample_times = np.linspace(0, clip.duration, num_samples)

        for i, t in enumerate(sample_times):
            try:
                print(f"    ⏳ Processing frame {i+1}/{num_samples} at {t:.2f}s...")
                frame = clip.get_frame(t)
                faces = self.detect_faces_in_frame(frame, frame_time=t)

                if faces:
                    best_face = faces[0]
                    face_positions.append(best_face['center_x'])
                    print(f"    ✅ Frame {i+1}: Found face at position {best_face['center_x']} with confidence {best_face['confidence']:.2f}")
                else:
                    if face_positions:
                        face_positions.append(face_positions[-1])
                    else:
                        face_positions.append(width // 2)
                    print(f"    ⚠️ Frame {i+1}: No faces detected, using fallback position")

            except Exception as e:
                print(f"    ⚠️ Error processing frame {i+1} at {t:.2f}s: {e}")
                if face_positions:
                    face_positions.append(face_positions[-1])
                else:
                    face_positions.append(width // 2)

        if face_positions:
            print("    ⏳ Calculating optimal tracking trajectory...")
            positions = [(pos, height // 2) for pos in face_positions]
            smoothed_positions = self.smooth_trajectory(positions, window_size=3)
            center_x = int(np.median([pos[0] for pos in smoothed_positions]))
            print(f"    ✅ Face tracking complete, optimal center: {center_x}")
        else:
            center_x = width // 2
            print("    ⚠️  No faces detected, using center crop")

        center_x = max(target_width // 2, min(width - target_width // 2, center_x))
        left = center_x - target_width // 2
        
        print(f"    ⏳ Cropping video to {target_width}x{height} (9:16 ratio) at x-position: {left}")
        
        self.face_cache = {}
        
        cropped_clip = clip.crop(x1=left, width=target_width)
        print(f"    ✅ Video cropping complete: {target_width}x{height}")
        return cropped_clip

    def close(self):
        """Releases resources used by the face detector."""
        try:
            self.face_cache = {}
            if hasattr(self, 'detector'):
                self.detector.close()
            print("🎯 Face tracking resources released")
        except Exception as e:
            print(f"⚠️ Error closing face tracker: {e}")
