# Wifey Mood (Mood Detection from Camera Feed)

This project captures frames from a camera feed, sends them to a Language Learning Model (LLM) for mood detection, and displays the results on the video feed.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Configuration](#configuration)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/edwin100394/mood-detection.git
    cd mood-detection
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Set up the environment variables in a `.env` file:
    ```dotenv
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_BASE_URL=your_openai_base_url
    VIDEO_SOURCE=your_video_source
    ```

2. Run the main script to start the camera feed and mood detection:
    ```sh
    python src/main.py
    ```

3. Press `q` to quit the camera feed.

## Testing

1. To run the tests, use the following command:
    ```sh
    pytest
    ```

## Configuration

- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_BASE_URL`: The base URL for the OpenAI API.
- `VIDEO_SOURCE`: The video source for the camera feed (e.g., `0` for the default camera, `1` for an external camera).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.