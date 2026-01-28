from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, CompositeVideoClip

from styles.caption_styles import CAPTION_STYLES, HIGHLIGHT_KEYWORDS


class CaptionMaker:
    """
    Creates and adds word-by-word captions to a video clip.
    """
    def __init__(self, selected_style='clean_white'):
        """
        Initializes the CaptionMaker with a selected caption style.

        Args:
            selected_style (str, optional): The style of captions to use.
                                            Defaults to 'clean_white'.
        """
        self.font_paths = self.find_available_fonts()
        self.selected_style = selected_style
        self.styles = CAPTION_STYLES
        self.highlight_keywords = HIGHLIGHT_KEYWORDS

    def find_available_fonts(self):
        """
        Finds available fonts on the system.

        Returns:
            dict: A dictionary of available font paths.
        """
        font_collections = {
            'bold': [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
                '/System/Library/Fonts/Helvetica.ttc',
                '/System/Library/Fonts/Arial.ttf',
                '/Windows/Fonts/arialbd.ttf',
                '/Windows/Fonts/calibrib.ttf',
                '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf'
            ],
            'regular': [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/System/Library/Fonts/Helvetica.ttc',
                '/Windows/Fonts/arial.ttf',
                '/Windows/Fonts/calibri.ttf',
                '/usr/share/fonts/TTF/DejaVuSans.ttf'
            ]
        }

        found_fonts = {'bold': None, 'regular': None}

        for font_type, paths in font_collections.items():
            for path in paths:
                if Path(path).exists():
                    found_fonts[font_type] = path
                    print(f"    📝 Found {font_type} font: {Path(path).name}")
                    break

        if not found_fonts['bold'] and not found_fonts['regular']:
            print("    📝 Using default system fonts")

        return found_fonts

    def get_font(self, font_type, font_size):
        """
        Gets a font object for a given type and size.

        Args:
            font_type (str): The type of font (e.g., 'bold', 'regular').
            font_size (int): The size of the font.

        Returns:
            ImageFont: An ImageFont object.
        """
        try:
            font_path = self.font_paths.get(font_type)
            if font_path:
                return ImageFont.truetype(font_path, font_size)
            else:
                return ImageFont.load_default()
        except:
            return ImageFont.load_default()

    def create_word_image(self, word, video_size, font_size, is_highlighted=False):
        """
        Creates an image of a single word.

        Args:
            word (str): The word to create an image of.
            video_size (tuple): The size of the video (width, height).
            font_size (int): The size of the font.
            is_highlighted (bool, optional): Whether the word should be
                                             highlighted. Defaults to False.

        Returns:
            numpy.ndarray: A numpy array representing the image.
        """
        width, height = video_size

        style_config = self.styles.get(self.selected_style, self.styles['clean_white'])
        font_type = style_config['font_type']

        font = self.get_font(font_type, font_size)

        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        temp_bbox = draw.textbbox((0, 0), word, font=font)
        text_width = temp_bbox[2] - temp_bbox[0]
        text_height = temp_bbox[3] - temp_bbox[1]

        x = (width - text_width) // 2

        safe_bottom_margin = int(height * 0.15)
        y = height - text_height - safe_bottom_margin

        if y < height * 0.7:
            y = int(height * 0.7)

        if is_highlighted:
            text_color = (255, 255, 0, 255)
        else:
            text_color = style_config['text_color']

        draw.text((x, y), word, font=font, fill=text_color)

        return np.array(img)

    def add_captions(self, clip, words, clip_start_time):
        """
        Adds word-by-word captions to a video clip.

        Args:
            clip (moviepy.editor.VideoFileClip): The video clip to add captions to.
            words (list): A list of words with timestamps.
            clip_start_time (float): The start time of the clip in the original
                                     video.

        Returns:
            moviepy.editor.VideoFileClip: The video clip with captions.
        """
        if not words:
            return clip

        caption_clips = []
        
        # Group words into phrases/sentences
        phrases = []
        current_phrase = []
        current_start = -1
        last_end = -1
        
        # Logic to group words: 
        # Breaks on punctuation, or if pause > 0.5s, or if phrase length > 6 words
        for i, word in enumerate(words):
            text = word['word'].strip()
            word_start = word['start'] - clip_start_time
            word_end = word['end'] - clip_start_time
            
            # Skip words outside clip
            if word_end <= 0 or word_start >= clip.duration:
                continue
                
            word_start = max(0, word_start)
            word_end = min(clip.duration, word_end)

            if not current_phrase:
                current_start = word_start
                current_phrase.append(text)
                last_end = word_end
            else:
                # Check for gap
                time_gap = word_start - last_end
                
                # Check for punctuation in previous word
                prev_text = current_phrase[-1]
                has_punctuation = prev_text.endswith(('.', '?', '!'))
                
                if time_gap > 0.5 or len(current_phrase) >= 6 or has_punctuation:
                    # End phrase
                    phrases.append({
                        'text': " ".join(current_phrase),
                        'start': current_start,
                        'end': last_end, # End at end of last word
                        'duration': last_end - current_start
                    })
                    # Start new phrase
                    current_phrase = [text]
                    current_start = word_start
                    last_end = word_end
                else:
                    current_phrase.append(text)
                    last_end = word_end
        
        # Add last phrase
        if current_phrase:
            phrases.append({
                'text': " ".join(current_phrase),
                'start': current_start,
                'end': last_end,
                'duration': last_end - current_start
            })

        video_width, video_height = clip.size
        base_font_size = max(40, int(min(video_width, video_height) * 0.05)) # Slightly smaller for phrases

        for phrase in phrases:
            text = phrase['text']
            start_time = phrase['start']
            # Extend duration slightly to cover gap to next phrase, for smoothness
            duration = phrase['duration']
            
            # Create solid image for the phrase
            text_img = self.create_word_image(text, (video_width, video_height), base_font_size, is_highlighted=False)
            
            # Create solid clip (no fade)
            txt_clip = ImageClip(text_img, duration=duration, transparent=True)
            txt_clip = txt_clip.set_start(start_time)
            
            caption_clips.append(txt_clip)

        if caption_clips:
            return CompositeVideoClip([clip] + caption_clips)
        return clip
