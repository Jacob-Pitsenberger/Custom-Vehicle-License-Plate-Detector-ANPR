# Custom Vehicle License Plate Detector

This project implements a custom vehicle license plate detector using a YOLO (You Only Look Once) object detection model. It is capable of detecting license plates in images, videos, and live webcam feeds. Additionally, Optical Character Recognition (OCR) is applied to extract the plate numbers from detected plates.

## Usage

### Training the Model
1. Download the dataset from [Roboflow](https://universe.roboflow.com/samrat-sahoo/license-plates-f8vsn).
2. Organize the dataset in the required structure in a directory titled `data`.
3. Create a `config.yaml` file specifying the paths to the training data in the `data` directory.
4. Train the model for 100 epochs using the `train.py` script. Transfer learning is utilized by starting with a pretrained YOLO model.

### Running the Detector
- To run object detection on an image, use `detect_on_img(img_path)`.
- To perform detection on a video, use `detect_on_video(video_path)`.
- For live detection on a webcam feed, use `detect_on_webcam()`.

### Main Entry Point
The `main()` function in `predict.py` serves as the entry point for the application. It specifies paths for input images and videos, and calls the necessary functions for object detection.

## Project Structure
- `data/`: Directory for storing the formatted training dataset.
- `images/`: Original images for testing the detector.
- `videos/`: Original videos for testing the detector.
- `runs/`: Directory where trained models are stored.
  - `detect/`: Subdirectory for detection-related files.
    - `train/`: Trained model weights are stored here (`best.pt`).
- `license_detections_webcam/`: Detected license plates from the webcam feed.
- `license_detections_image/`: Detected license plates from images.
- `license_detections_video/`: Detected license plates from videos.

## Transfer Learning
This model employs transfer learning by utilizing a pretrained YOLO model. This approach allows the model to leverage knowledge gained from training on a large dataset for a similar task, significantly reducing the time and data required to achieve good performance on the license plate detection task.

## Dependencies
- [Ultralytics](https://github.com/ultralytics/yolov5)
- OpenCV
- EasyOCR
- NumPy

## References
- Source of `filter_text` and `save_results` functions:
  [RealTimeAutomaticNumberPlateRecognition](https://github.com/nicknochnack/RealTimeAutomaticNumberPlateRecognition/blob/main/Automatic%20Number%20Plate%20Detection.ipynb)

## Author
Jacob Pitsenberger

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.