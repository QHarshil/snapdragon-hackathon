import cv2
import numpy as np
import logging
import os

# Import the SNPE Python API
try:
    from snpe import SNPE
except ImportError as e:
    raise ImportError("Qualcomm SNPE API not found. Make sure the SNPE SDK is installed.") from e


class VisionProcessor:
    """
    Captures frames from a camera and uses Qualcomm's SNPE APIs to perform object detection.
    """

    def __init__(self, dlc_file: str, runtime: str = "GPU", camera_index: int = 0):
        """
        Initializes the VisionProcessor using a Qualcomm SNPE DLC file.

        Args:
            dlc_file (str): Path to the SNPE DLC file.
            runtime (str): Runtime to use (e.g., 'GPU', 'CPU', 'DSP').
            camera_index (int): Index of the camera device.
        """
        self.logger = logging.getLogger(__name__)
        self.camera_index = camera_index

        # Initialize camera capture.
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera at index {camera_index}")
        self.logger.info("Camera opened successfully on index %d.", camera_index)

        # Check if the DLC file exists.
        if not os.path.exists(dlc_file):
            raise ValueError(f"DLC file not found: {dlc_file}")

        # Initialize SNPE with the specified runtime.
        self.snpe = SNPE(runtime=runtime)
        self.snpe.load_dlc(dlc_file)
        self.logger.info("SNPE loaded DLC file: %s with runtime: %s", dlc_file, runtime)

        # Retrieve the input tensor shape from the SNPE model.
        # (Assuming SNPE API provides a method get_input_shape() that returns a shape like [1, height, width, channels])
        self.input_shape = self.snpe.get_input_shape()
        self.logger.info("SNPE model input shape: %s", self.input_shape)

    def capture_frame(self):
        """
        Captures a frame from the camera.

        Returns:
            numpy.ndarray: The captured frame or None if capture fails.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to capture frame from camera.")
            return None
        self.logger.debug("Captured frame with shape: %s", frame.shape)
        return frame

    def preprocess_frame(self, frame):
        """
        Preprocesses the frame for SNPE inference.

        Args:
            frame (numpy.ndarray): The raw captured frame.

        Returns:
            numpy.ndarray: Preprocessed input tensor.
        """
        try:
            # Assume self.input_shape is of the form [1, height, width, channels].
            _, height, width, _ = self.input_shape
            # Resize frame to match the expected input dimensions.
            frame_resized = cv2.resize(frame, (width, height))
            # Normalize pixel values to the [0, 1] range.
            frame_normalized = frame_resized.astype('float32') / 255.0
            # Expand dimensions to match the input tensor shape.
            input_tensor = np.expand_dims(frame_normalized, axis=0)
            self.logger.debug("Preprocessed frame to shape: %s", input_tensor.shape)
            return input_tensor
        except Exception as e:
            self.logger.error("Error preprocessing frame: %s", e)
            raise

    def detect_objects(self, frame, score_threshold: float = 0.5):
        """
        Runs SNPE inference on the provided frame to detect objects.

        Args:
            frame (numpy.ndarray): The raw captured frame.
            score_threshold (float): Minimum confidence score for valid detections.

        Returns:
            list: A list of detections. Each detection is a dict with keys 'box', 'class', and 'score'.
        """
        try:
            input_tensor = self.preprocess_frame(frame)
            # Run inference using SNPE.
            # (Assuming the SNPE API provides a method run_inference that returns a dictionary of outputs.)
            outputs = self.snpe.run_inference(input_tensor)
            # Example: the model outputs might include "detection_boxes", "detection_classes", and "detection_scores".
            boxes = outputs.get("detection_boxes", [])
            classes = outputs.get("detection_classes", [])
            scores = outputs.get("detection_scores", [])

            detections = []
            for i in range(len(scores)):
                if scores[i] >= score_threshold:
                    detection = {
                        'box': boxes[i],
                        'class': int(classes[i]),
                        'score': float(scores[i])
                    }
                    detections.append(detection)
            self.logger.debug("Detected %d objects with threshold %.2f", len(detections), score_threshold)
            return detections
        except Exception as e:
            self.logger.error("Error during SNPE object detection: %s", e)
            return []

    def close(self):
        """
        Releases the camera resource.
        """
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Camera resource released.")


# Standalone testing for VisionProcessor using SNPE.
if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.DEBUG)
    # Specify the path to the DLC file and desired runtime.
    dlc_file = "model.dlc"
    vision_processor = VisionProcessor(dlc_file=dlc_file, runtime="GPU", camera_index=0)
    try:
        start_time = time.time()
        while time.time() - start_time < 10:
            frame = vision_processor.capture_frame()
            if frame is not None:
                detections = vision_processor.detect_objects(frame)
                print("Detections:", detections)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        vision_processor.close()
