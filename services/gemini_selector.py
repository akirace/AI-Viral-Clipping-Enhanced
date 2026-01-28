import json
import random
import google.generativeai as genai
from config import GEMINI_MODEL

class GeminiSelector:
    """
    Uses the Gemini AI model to select the most viral clips from a transcript.
    """
    def __init__(self, api_key):
        """
        Initializes the GeminiSelector with an API key.

        Args:
            api_key (str): The API key for the Gemini AI model.
        """
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        try:
            print(f"🤖 Initializing Gemini model: {GEMINI_MODEL}")
            self.model = genai.GenerativeModel(GEMINI_MODEL)
        except Exception as e:
            print(f"⚠️ Failed to initialize {GEMINI_MODEL}: {e}")
            print("    Using fallback model: gemini-pro")
            self.model = genai.GenerativeModel('gemini-pro')

    def select_clips(self, segments, video_duration, n, min_dur, max_dur):
        """
        Selects the most viral clips from a transcript using the Gemini AI model.

        Args:
            segments (list): A list of transcript segments with timestamps.
            video_duration (float): The total duration of the video.
            n (int): The number of clips to select.
            min_dur (int): The minimum duration of each clip.
            max_dur (int): The maximum duration of each clip.

        Returns:
            list: A list of dictionaries, each representing a selected clip.
        """
        segments_text = []
        for i, seg in enumerate(segments):
            segments_text.append(f"[{seg['start']:.1f}s-{seg['end']:.1f}s]: {seg['text']}")
        
        transcript_with_timestamps = "\n".join(segments_text)
        
        if n == 0:
            count_instruction = "ALL viral-worthy clips"
            limit_instruction = "Select as many clips as are truly viral-worthy."
        else:
            count_instruction = f"the {n} BEST viral clips"
            limit_instruction = f"Select exactly {n} clips."

        prompt = f"""You are an expert at creating viral short-form content like Opus.pro. Analyze this transcript with precise timestamps and select {count_instruction}.
{limit_instruction}

CRITICAL RULES:
1. Each clip MUST start at the EXACT beginning of a sentence/thought and end at the EXACT completion of that sentence/thought
2. Never cut off mid-sentence or mid-word - clips must be complete thoughts
3. Each clip must be AT LEAST {min_dur} seconds long (strict lower limit). Max {max_dur} seconds.
4. Clips cannot overlap and must use the EXACT timestamps provided
5. Focus on complete viral moments: hooks, revelations, advice, stories, funny moments
6. DO NOT select clips shorter than {min_dur} seconds under any circumstances. Combine adjacent segments if necessary to meet the duration.

SELECTION CRITERIA (prioritize):
- Complete engaging stories or thoughts
- Surprising facts or revelations 
- Actionable advice or tips
- Emotional moments or reactions
- Quotable one-liners with context
- Question-answer pairs

VIDEO DURATION: {video_duration} seconds

TRANSCRIPT WITH EXACT TIMESTAMPS:
{transcript_with_timestamps}

Return ONLY valid JSON with EXACT timestamps from the transcript:
{{
  "clips": [
    {{
      "start": 34.5,
      "end": 67.2,
      "title": "Complete thought or hook",
      "virality_score": 85,
      "hook_type": "story_reveal",
      "reason": "Complete engaging story with clear beginning and end"
    }}
  ]
}}"""
        
        try:
            print("🤖 AI analyzing transcript for complete viral thoughts...")
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            data = json.loads(response.text)
            validated_clips = []
            
            for clip_data in data.get('clips', []):
                start = clip_data.get('start')
                end = clip_data.get('end')
                title = clip_data.get('title', 'Untitled')
                score = clip_data.get('virality_score', 0)
                hook_type = clip_data.get('hook_type', 'general')

                if start is None or end is None:
                    continue

                start, end = float(start), float(end)
                duration = end - start
                
                if duration > max_dur:
                    end = start + max_dur
                    duration = max_dur
                    
                if min_dur <= duration <= max_dur and start < end and end <= video_duration:
                    validated_clips.append({
                        'start': start,
                        'end': end,
                        'title': title,
                        'virality_score': score,
                        'hook_type': hook_type,
                        'duration': duration
                    })

            if not validated_clips:
                raise ValueError("AI did not return any valid clips.")

            validated_clips.sort(key=lambda x: x['virality_score'], reverse=True)
            
            # Slice if n > 0, otherwise return all
            final_clips = validated_clips[:n] if n > 0 else validated_clips
            
            print(f"✅ AI selected {len(final_clips)} complete viral clips:")
            for i, clip in enumerate(final_clips, 1):
                print(f"  {i}. {clip['title']} (Score: {clip['virality_score']}, Type: {clip['hook_type']})")
            
            return final_clips
            
        except Exception as e:
            print(f"❌ AI clip selection failed: {e}. Using fallback method.")
            return self._fallback_selection(segments, video_duration, n, min_dur, max_dur)

    def _fallback_selection(self, segments, video_duration, n, min_dur, max_dur):
        """
        Fallback method to select clips random-ish but ensuring minimum duration.
        """
        clips = []
        used_segments = set()
        
        # Try to find n clips
        for _ in range(n):
            available_indices = [i for i in range(len(segments)) if i not in used_segments]
            if not available_indices:
                break
                
            # Pick a random starting segment
            start_idx = random.choice(available_indices)
            
            current_idx = start_idx
            current_duration = 0
            merged_text = ""
            start_time = segments[start_idx]['start']
            end_time = segments[start_idx]['end']
            
            # Merge segments forward until we hit min_dur or max_dur
            while current_idx < len(segments):
                if current_idx in used_segments and current_idx != start_idx:
                    # Hit a used segment, stop expanding
                    break
                    
                seg = segments[current_idx]
                new_duration = seg['end'] - start_time
                
                if new_duration > max_dur:
                    # Too long, stop before adding this one (or if it's the first one, just take it clipped)
                    if current_duration == 0:
                        end_time = start_time + max_dur
                        current_duration = max_dur
                        used_segments.add(current_idx)
                    break
                
                # Add this segment
                used_segments.add(current_idx)
                end_time = seg['end']
                current_duration = new_duration
                merged_text += seg['text'] + " "
                current_idx += 1
                
                # If we met the minimum duration, we can consider stopping
                # But let's verify if we can add more context up to max_dur? 
                # For fallback, just hitting min_dur is good enough to prevent checking too much
                if current_duration >= min_dur:
                    break
            
            # Only add if it meets reasonable criteria (or if it's the best we could do)
            if current_duration >= min_dur or (current_duration > 5 and len(segments) < 5):
                 clips.append({
                    'start': start_time,
                    'end': end_time,
                    'title': f'Fallback clip {len(clips)+1}',
                    'virality_score': 50,
                    'hook_type': 'general',
                    'duration': current_duration
                })

        return clips
