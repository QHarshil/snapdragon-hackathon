import json
import logging

import pyaudio
import pyttsx3
from vosk import Model, KaldiRecognizer


class AudioProcessor:
    """
    Handles capturing audio input, converting speech to text using Vosk,
    and synthesizing speech output with pyttsx3.
    """

    def __init__(self, model_path: str = 'model', rate: int = 16000, chunk_size: int = 4000):
        """
        Initializes the AudioProcessor.

        Args:
            model_path (str): Path to the Vosk model directory.
            rate (int): Audio sampling rate.
            chunk_size (int): Number of frames to read per capture.
        """
        self.logger = logging.getLogger(__name__)
        self.rate = rate
        self.chunk_size = chunk_size

        try:
            self.engine = pyttsx3.init()
            self.logger.info("TTS engine initialized.")
        except Exception as e:
            self.logger.error("Failed to initialize TTS engine: %s", e)
            raise

        try:
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, self.rate)
            self.logger.info("Vosk model loaded from: %s", model_path)
        except Exception as e:
            self.logger.error("Error loading Vosk model from '%s': %s", model_path, e)
            raise

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.stream.start_stream()
            self.logger.info("Audio stream started successfully.")
        except Exception as e:
            self.logger.error("Error initializing audio stream: %s", e)
            raise

    def capture_audio(self) -> bytes:
        """
        Captures a chunk of audio data.

        Returns:
            bytes: Captured audio data.
        """
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            self.logger.debug("Captured audio data of length: %d", len(data))
            return data
        except Exception as e:
            self.logger.error("Error capturing audio: %s", e)
            return b""

    def process_stt(self, audio_data: bytes) -> str:
        """
        Processes audio data to extract text using Vosk.

        Args:
            audio_data (bytes): Raw audio data.

        Returns:
            str: Recognized text.
        """
        if not audio_data:
            self.logger.warning("No audio data provided for STT processing.")
            return ""

        try:
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                self.logger.debug("STT recognized: %s", text)
                return text
            else:
                self.logger.debug("Incomplete audio input; no final STT result yet.")
                return ""
        except Exception as e:
            self.logger.error("Error during STT processing: %s", e)
            return ""

    def speak(self, text: str) -> None:
        """
        Converts text to speech and outputs it.

        Args:
            text (str): Text to be spoken.
        """
        if not text:
            self.logger.warning("No text provided for TTS synthesis.")
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
            self.logger.debug("Spoken text: %s", text)
        except Exception as e:
            self.logger.error("Error during TTS synthesis: %s", e)

    def close(self) -> None:
        """
        Closes the audio stream and terminates PyAudio.
        """
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
                self.logger.info("Audio stream closed.")
            if self.pyaudio_instance is not None:
                self.pyaudio_instance.terminate()
                self.logger.info("PyAudio instance terminated.")
        except Exception as e:
            self.logger.error("Error closing audio resources: %s", e)


# Standalone testing.
if __name__ == '__main__':
    import time
    logging.basicConfig(level=logging.DEBUG)
    audio_processor = None
    try:
        audio_processor = AudioProcessor(model_path="model")
        start_time = time.time()
        while time.time() - start_time < 10:
            audio_chunk = audio_processor.capture_audio()
            recognized_text = audio_processor.process_stt(audio_chunk)
            if recognized_text:
                print("Recognized:", recognized_text)
                audio_processor.speak("I heard " + recognized_text)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as ex:
        logging.error("Unexpected error: %s", ex)
    finally:
        if audio_processor is not None:
            audio_processor.close()
