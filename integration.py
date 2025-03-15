import logging


class DecisionEngine:
    """
    Processes inputs from the audio and vision modules to determine if a warning
    should be issued.
    """

    def __init__(self, obstacle_classes=None):
        """
        Initializes the DecisionEngine.

        Args:
            obstacle_classes (list, optional): List of class IDs representing obstacles.
                                               Defaults to [1, 2, 3] if not provided.
        """
        self.logger = logging.getLogger(__name__)
        self.obstacle_classes = obstacle_classes if obstacle_classes is not None else [1, 2, 3]
        self.logger.info("DecisionEngine initialized with obstacle classes: %s", self.obstacle_classes)

    def process_inputs(self, audio_text: str, detection_results: list) -> str:
        """
        Combines STT text and object detection results to determine if an alert is needed.

        Args:
            audio_text (str): Recognized text from the audio module.
            detection_results (list): List of detections from VisionProcessor.

        Returns:
            str: Warning message if an obstacle is detected; otherwise, an empty string.
        """
        try:
            obstacle_detected = any(
                detection.get('class') in self.obstacle_classes for detection in detection_results
            )
            if obstacle_detected:
                warning_message = "Warning: Obstacle ahead!"
                self.logger.debug("Obstacle detected. Issuing warning: %s", warning_message)
                return warning_message
            self.logger.debug("No obstacles detected.")
            return ""
        except Exception as e:
            self.logger.error("Error processing inputs: %s", e)
            return ""


# Standalone testing.
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    sample_audio_text = "Proceed forward"
    sample_detections = [
        {'box': [0.1, 0.2, 0.3, 0.4], 'class': 1, 'score': 0.75},
        {'box': [0.5, 0.6, 0.7, 0.8], 'class': 5, 'score': 0.65},
    ]
    decision_engine = DecisionEngine()
    result = decision_engine.process_inputs(sample_audio_text, sample_detections)
    print("DecisionEngine output:", result)
