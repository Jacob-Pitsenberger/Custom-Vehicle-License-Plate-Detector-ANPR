"""
predict.py

This module contains functions for object detection using a custom YOLO model, along with supporting functions
for processing images, videos, and webcam feeds. It also includes utility functions for saving results and filtering
detected text.

Author: Jacob Pitsenberger
Date: 9/19/23

Usage:
    - To run object detection on an image, call detect_on_img(img_path).
    - To run object detection on a video, call detect_on_video(video_path).
    - To perform object detection on a live webcam feed, call detect_on_webcam().
    - The main() function provides an entry point for running the application.

References:
    - Source of filter_text and save_results functions:
      https://github.com/nicknochnack/RealTimeAutomaticNumberPlateRecognition/blob/main/Automatic%20Number%20Plate%20Detection.ipynb
"""

from ultralytics import YOLO
import os
import cv2
import easyocr
import numpy as np
import csv
import uuid

def detect_on_video(video_path) -> None:
    """
    Detects objects in a video using a custom YOLO model and saves the annotated video.

    This function takes a video file, processes each frame using a YOLO model, and detects objects.
    Detected objects are annotated with bounding boxes and class names. The annotated video is then saved.

    Args:
        video_path (str): The path to the input video file.

    Returns:
        None
    """
    # Specify the path to save the video with found detections.
    video_path_out = '{}_prediction.mp4'.format(video_path)

    # Create a video capture object for the video to predict upon.
    cap = cv2.VideoCapture(video_path)

    # Start reading the video.
    ret, frame = cap.read()

    # Get the dimensions of the video frames.
    H, W, _ = frame.shape

    # Initialize our video writer for saving the output video.
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    # Specify the path to the custom model.
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

    # Load the model from the model path
    model = YOLO(model_path)

    # Define the threshold for good detections.
    threshold = 0.7

    # Loop through frames read from the video file.
    while ret:

        # Detect with our custom model over each frame.
        results = model(frame)[0]

        # Loop through found detections.
        for result in results.boxes.data.tolist():
            # Get the bounding box coordinates, detection scores, and class id's of found detections.
            x1, y1, x2, y2, score, class_id = result

            # Define bounding boxes for detections.
            region = frame[int(y1):int(y2), int(x1):int(x2)]

            # If the score is better than our threshold for good detections...
            if score > threshold:

                # and only if the detection is a license plate
                if int(class_id) == 0:
                    try:
                        # Create a Reader and have it read the text on the license plate.
                        reader = easyocr.Reader(['en'])
                        ocr_result = reader.readtext(region)

                        # Get the plate number only using the filter_text function.
                        text = filter_text(region, ocr_result, 0.5)

                        # Draw the text (plate number detected) on the license plate image
                        cv2.putText(region, f"{text}", (int(x1) + 25, int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                        # Save the license plate image and the detected text.
                        save_results(text, region, 'license_detection_results_video.csv', 'license_detections_video')
                    except:
                        pass
                # For all detections, draw the bounding box and class name around the detection on the original image.
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        # Write the frame to the output file.
        out.write(frame)
        # Keep reading the frames from the video file until they have all been processed.
        ret, frame = cap.read()

    # Release resources when the video is done being processed.
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def detect_on_img(img_path) -> None:
    """
    Detects objects in an image using a custom YOLO model.

    This function loads a custom YOLO model, sets a threshold for considering detections,
    processes the provided image, and detects objects. If a license plate is detected,
    it performs Optical Character Recognition (OCR) to extract the plate number and overlays
    it on the image. Detected vehicles are annotated with bounding boxes and class names.

    Args:
        img_path (str): The path to the input image file.

    Returns:
        None
    """
    # Specify the path to save the image with found detections.
    img_path_out = '{}_prediction.png'.format(img_path)

    # Specify the path to the custom model.
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

    # Load the model from the model path
    model = YOLO(model_path)

    # Define the threshold for good detections.
    threshold = 0.7

    # Read the image.
    img = cv2.imread(img_path)

    # Make detections on the image.
    results = model(img)[0]

    # Loop through found detections.
    for result in results.boxes.data.tolist():
        # Get the bounding box coordinates, detection scores, and class id's of found detections.
        x1, y1, x2, y2, score, class_id = result

        # Define bounding boxes for detections.
        region = img[int(y1):int(y2), int(x1):int(x2)]

        # If the score is better than our threshold for good detections...
        if score > threshold:

            # and only if the detection is a license plate
            if int(class_id) == 0:
                try:
                    # Create a Reader and have it read the text on the license plate.
                    reader = easyocr.Reader(['en'])
                    ocr_result = reader.readtext(region)

                    # Get the plate number only using the filter_text function.
                    text = filter_text(region, ocr_result, 0.5)

                    # Draw the text (plate number detected) on the license plate image
                    cv2.putText(region, f"{text}", (int(x1) + 25, int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    # Save the license plate image and the detected text.
                    save_results(text, region, 'license_detection_results_image.csv', 'license_detections_image')
                except:
                    pass
            # For all detections, draw the bounding box and class name around the detection on the original image.
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1, cv2.LINE_AA)

    # Resize this image for easier viewing.
    img = cv2.resize(img, (720, 480))
    # Show the original image with all detections found using the custom model.
    while True:
        cv2.imshow('out', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite(img_path_out, img)
    cv2.destroyAllWindows()

def detect_on_webcam() -> None:
    """
    Performs object detection on live webcam feed using a custom YOLO model.

    This function captures video from the computer's webcam, processes each frame using a YOLO model,
    and detects objects. If a license plate is detected, it performs Optical Character Recognition (OCR)
    to extract the plate number and overlays it on the image. Detected vehicles are annotated with bounding
    boxes and class names.

    Args:
        None

    Returns:
        None
    """
    # Create video capture object for computers webcam.
    cap = cv2.VideoCapture(0)

    # Read frames from the webcam.
    ret, frame = cap.read()

    # Get the dimensions of the webcams video frames.
    H, W, _ = frame.shape

    # Specify the path to the custom model.
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

    # Load the model from the model path
    model = YOLO(model_path)

    # Define the threshold for good detections.
    threshold = 0.8

    # Loop through frames read from the webcam.
    while ret:
        # Make detections on the image.
        results = model(frame)[0]

        # Loop through found detections.
        for result in results.boxes.data.tolist():
            # Get the bounding box coordinates, detection scores, and class id's of found detections.
            x1, y1, x2, y2, score, class_id = result

            # Define bounding boxes for detections.
            region = frame[int(y1):int(y2), int(x1):int(x2)]

            # If the score is better than our threshold for good detections...
            if score > threshold:

                # and only if the detection is a license plate
                if int(class_id) == 0:
                    try:
                        # Create a Reader and have it read the text on the license plate.
                        reader = easyocr.Reader(['en'])
                        ocr_result = reader.readtext(region)

                        # Get the plate number only using the filter_text function.
                        text = filter_text(region, ocr_result, 0.5)

                        # Draw the text (plate number detected) on the license plate image
                        cv2.putText(region, f"{text}", (int(x1) + 25, int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .25, (0, 0, 255), 1, cv2.LINE_AA)

                        # Save the license plate image and the detected text.
                        save_results(text, region, 'license_detection_results_webcam.csv', 'license_detections_webcam')
                    except:
                        pass
                else:
                    # For vehicle detections, draw the bounding box and class name around the detection on the original image.
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                    cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0), 1, cv2.LINE_AA)
        # Show the current frame.
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            # Continue reading frames from the webcam.
            ret, frame = cap.read()
    # Release the video capture object and destroy the window.
    cap.release()
    cv2.destroyAllWindows()


def filter_text(region: np.ndarray, ocr_result: list, region_threshold: float) -> list:
    """This function is 100% from the source: https://github.com/nicknochnack/RealTimeAutomaticNumberPlateRecognition/blob/main/Automatic%20Number%20Plate%20Detection.ipynb

    Filters small non-plate-number text from OCR results.

    Args:
        region (numpy.ndarray): The region of interest (license plate image).
        ocr_result (list): The OCR results for the region.
        region_threshold (float): The threshold for considering text as plate number.

    Returns:
        list: A list of filtered plate numbers.
    """
    # Calculate the size (area) of the region of interest
    rectangle_size = region.shape[0] * region.shape[1]

    # Initialize an empty list called plate to store filtered plate numbers.
    plate = []
    # Iterate through each OCR result in the list of results.
    for result in ocr_result:
        # Calculate the length and height of the bounding box of the detected text.
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        # Check if the area of the bounding box is greater than the specified threshold.
        if length * height / rectangle_size > region_threshold:
            # If the text meets the criteria, it is added to the list of filtered plate numbers.
            plate.append(result[1])

    # Return the list of filtered plate numbers.
    return plate


def save_results(text: list, region: np.ndarray, csv_filename: str, folder_path: str) -> None:
    """This function is 100% from the source: https://github.com/nicknochnack/RealTimeAutomaticNumberPlateRecognition/blob/main/Automatic%20Number%20Plate%20Detection.ipynb

    Saves the detected text and region of interest to a CSV file and as an image file.

    Args:
        text (list): The detected text (license plate number).
        region (numpy.ndarray): The region of interest (license plate image).
        csv_filename (str): The name of the CSV file for storing results.
        folder_path (str): The path to the folder for storing images.

    Returns:
        None

    """
    # Generate a unique image name using the uuid module.
    img_name = '{}.jpg'.format(uuid.uuid1())

    # Save the image of the license plate.
    cv2.imwrite(os.path.join(folder_path, img_name), region)

    # Open the csv file and append newly detected license plate numbers and the image name for them to it.
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])

def main():
    """
    Entry point for the application.

    This function serves as the entry point for the application. It specifies paths for input images and videos,
    and calls the necessary functions to perform object detection on an image, a video, or through a webcam.

    Args:
        None

    Returns:
        None
    """
    # Path to the videos Directory.
    VIDEOS_DIR = os.path.join('.', 'videos')

    # Path to the test image and video files.
    img_path = 'images/cars.png'
    video_path = os.path.join(VIDEOS_DIR, 'cars_on_highway.mp4')

    # detect_on_img(img_path)
    # detect_on_video(video_path)
    detect_on_webcam()


if __name__ == "__main__":
    main()
