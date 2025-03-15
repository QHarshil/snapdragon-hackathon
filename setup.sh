echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setting up Vosk model..."
if [ ! -d "models/vosk" ]; then
    mkdir -p models/vosk
    echo "Downloading Vosk small English model..."
    wget -O models/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip models/vosk.zip -d models/
    mv models/vosk-model-small-en-us-0.15 models/vosk
    rm models/vosk.zip
else
    echo "Vosk model already exists."
fi

echo "Setting up TensorFlow Lite model..."
if [ ! -f "models/mobilenet_ssd.tflite" ]; then
    echo "Downloading MobileNet SSD TFLite model..."
    
    
    ###### placeholder for TFLite model.
    wget -O models/mobilenet_ssd.zip https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
    unzip models/mobilenet_ssd.zip -d models/
    # Move the .tflite file to the models folder and clean up.
    mv models/ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite models/mobilenet_ssd.tflite
    rm -rf models/ssd_mobilenet_v1_1.0_quant_2018_06_29 models/mobilenet_ssd.zip
else
    echo "TensorFlow Lite model already exists."
fi

echo "Setup complete!"
