import time
import logging
from audio import AudioProcessor
from vision import VisionProcessor
from integration import DecisionEngine

def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Starting integrated obstacle detection system with SNPE...")

    # Initialize modules.
    audio_processor = AudioProcessor(model_path='model', rate=16000, chunk_size=4000)
    # For SNPE, we use the DLC file (e.g. 'model.dlc'). Adjust the runtime as needed.
    vision_processor = VisionProcessor(dlc_file='model.dlc', runtime='GPU', camera_index=0)
    decision_engine = DecisionEngine(obstacle_classes=[1, 2, 3])  # Adjust obstacle class IDs as needed.

    try:
        while True:
            # Process audio.
            audio_data = audio_processor.capture_audio()
            recognized_text = audio_processor.process_stt(audio_data)
            if recognized_text:
                logging.debug("Recognized text: %s", recognized_text)

            # Process video.
            frame = vision_processor.capture_frame()
            if frame is None:
                continue
            detections = vision_processor.detect_objects(frame)
            logging.debug("Detections: %s", detections)

            # Decision logic: fuse audio and vision.
            warning_message = decision_engine.process_inputs(recognized_text, detections)
            if warning_message:
                logging.info("Issuing warning: %s", warning_message)
                audio_processor.speak(warning_message)

            # Small delay between iterations.
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Shutting down...")
    except Exception as e:
        logging.error("Unexpected error: %s", e)
    finally:
        audio_processor.close()
        vision_processor.close()

if __name__ == "__main__":
    main()
