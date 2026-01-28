# 🎬 Enhanced AI Clipping Software

> **Supercharged Version** of the original [LikithMeruvu/AI-Clipping-Software](https://github.com/LikithMeruvu/AI-Clipping-Software).

This repository contains a significantly enhanced version of the AI Clipping Software. It leverages **Google Gemini 2.5 Flash Lite**, **Faster-Whisper (CUDA)**, and **MediaPipe** to automatically detect, transcribe, and crop viral moments from long-form videos.

## 🚀 Key Enhancements

This fork includes major upgrades for performance, flexibility, and content quality:

### 1. 🧠 Advanced AI Selection
*   **Unlimited Auto-Mode**: Enter `0` or `auto` to let Gemini find **ALL** viral-worthy clips in a video, with no arbitrary limit.
*   **Upgraded Model**: Switched to the faster and sharper **`gemini-2.5-flash-lite`**.
*   **Strict Duration Control**: New logic ensures clips *never* fall below your minimum duration (e.g., 60s). It aggressively merges context to prevent short, useless 5-second clips.

### 2. ⚡ High-Performance Architecture
*   **Forced CUDA (GPU)**: Optimized specifically for NVIDIA GPUs (RTX series). It enforces CUDA usage for the Whisper transcriber, ensuring lightning-fast transcription speeds.
*   **Better Downloader**: Added a visual progress bar and **manual quality selection** (choose between 1080p, 720p, etc.) before downloading.

### 3. 🎯 Smart Region-Aware Face Tracking
*   **Multi-Pass Detection**:
    1.  **Fast Scan**: Checks a downscaled frame for speed.
    2.  **Full-Res Retry**: If that fails, retries on the original high-res image.
    3.  **Region Fallback**: Crucial for **Podcasts**. If generic detection fails, it specifically scans the **Left Half** and **Right Half** to lock onto a host, ensuring the camera never crops to an empty center.

### 4. 📂 Local Workflow
*   **Process Local Files**: You no longer need to re-download videos. Select any video file directly from the `/temp` folder to re-process it instantly.

### 5. 🎨 Professional Caption Styling
*   **Solid Phrase Captions**: Replaced the distracting "karaoke-style" single-word animation with **static, phrase-based captions**.
*   **Enhanced Readability**: Text appears in solid blocks (sentence-by-sentence) with high-contrast colors, matching the style of top-tier viral shorts.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/AI-Clipping-Software-Enhanced.git
    cd AI-Clipping-Software-Enhanced
    ```

2.  **Install Dependencies (CUDA Support Required):**
    *   First, ensure you have NVIDIA Drivers installed.
    *   Install PyTorch with CUDA 12.x support:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        ```
    *   Install the rest of the requirements:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Setup API Key:**
    *   Create a `.env` file and add your Google Gemini API key:
        ```env
        GEMINI_API_KEY=your_actual_api_key_here
        ```

## 🎮 Usage

Run the main script:
```bash
python main.py
```

You will be prompted to:
1.  **Source**: Enter a YouTube URL **OR** select a local file.
2.  **Clips**: Enter `0` for **Unlimited Auto-Mode** (recommended) or a specific number.
3.  **Duration**: Set min/max seconds (e.g., 50s - 60s).
4.  **Style**: Choose your caption style (e.g., Clean White, Bright Yellow).

---

## 📜 Credits

*   Original concept by [LikithMeruvu](https://github.com/LikithMeruvu).
*   Enhancements implemented by Dicky Arya.
