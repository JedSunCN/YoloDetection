import argparse
import csv
import os

import cv2
from PIL import Image
from ultralytics import YOLO

STORAGE_DIR = 'D:/STEBRIN'


class YOLOModel:
    def __init__(self, task_type, input_path, save_dir):
        super().__init__()
        self.task_type = task_type  # "image" or "video"
        self.input_path = input_path
        self.save_dir = save_dir

        model_path = os.path.join(STORAGE_DIR, 'model', 'model.pt')
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def detect_rust(self):
        if self.task_type == "image":
            return self.detect_rust_in_image(self.input_path, self.save_dir)
        elif self.task_type == "video":
            return self.detect_rust_in_video(self.input_path, self.save_dir)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def detect_rust_in_image(self, input_dir, save_dir):
        # Ensure the input directory exists
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory {input_dir} does not exist")

        # Create the save directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        total_corrosion_count = 0

        # Iterate over all image files in the input directory
        csv_output_path = os.path.join(save_dir, "检测结果.csv")
        with open(csv_output_path, mode='w', newline='', encoding='GBK') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["FileName", "CorrosionCount", "AbsolutePath"])

            # Iterate over all image files in the input directory
            for image_name in os.listdir(input_dir):
                image_path = os.path.join(input_dir, image_name)
                if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.png')):
                    results = self.model(image_path)

                    # Get the number of detected rust instances
                    corrosion_count = len(results[0].boxes)
                    total_corrosion_count += corrosion_count

                    for r in results:
                        im_array = r.plot()  # Draw the prediction results on a BGR numpy array
                        im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB PIL image

                        # Save the result image
                        base_name = os.path.splitext(image_name)[0]
                        result_name = f"{base_name}_result_corrnum_{corrosion_count}.jpeg"
                        result_path = os.path.join(save_dir, result_name)
                        im.save(result_path)

                        # Write the result to the CSV file
                        csv_writer.writerow([image_name, corrosion_count, os.path.abspath(result_path)])

        return total_corrosion_count  # Return the total number of detected rust instances

    def detect_rust_in_video(self, video_path, save_dir):
        # Ensure the input directory exists
        if not os.path.isdir(video_path):
            raise ValueError(f"Input directory {video_path} does not exist")

        # Create the save directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        output_paths = []

        # Iterate over all video files in the input directory
        for video_name in os.listdir(video_path):
            video_path = os.path.join(video_path, video_name)
            if os.path.isfile(video_path) and video_path.lower().endswith(('.mp4', '.avi', '.mov', '.flv')):
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                output_filename = f"{os.path.basename(video_path)}_result.mp4"
                output_path = os.path.join(save_dir, output_filename)
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    results = self.model(frame)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                cap.release()
                out.release()

                output_paths.append(output_path)

        return output_paths

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="YOLO Rust Detection")
    parser.add_argument("task_type", choices=["image", "video"], help="Type of task: 'image' or 'video'")
    parser.add_argument("input_path", help="Path to the input image or video file")
    parser.add_argument("save_dir", help="Directory to save the detection results")
    args = parser.parse_args()

    # Create YOLOModel instance
    yolo_model = YOLOModel(args.task_type, args.input_path, args.save_dir)

    # Perform rust detection
    result_path = yolo_model.detect_rust()

    # Print the result path
    print(f"Detection result saved at: {result_path}")

if __name__ == "__main__":
    main()