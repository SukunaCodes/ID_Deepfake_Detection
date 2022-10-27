import json
import os
import cv2
import math

base_path = '.\\Training Sample Videos\\'


def get_file_name(file_path):
    filename_basename = os.path.basename(file_path)
    filename_only = filename_basename.split('.')[0]
    return filename_only


with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

for filename in metadata.keys():
    print(filename)
    if filename.endswith('.mp4'):
        tmp_path = os.path.join(base_path, get_file_name(filename))
        print('Creating Directory😶‍🌫️😶‍🌫️....: ' + tmp_path)
        os.makedirs(tmp_path, exist_ok=True)
        print('Converting Video to Image😶‍🌫️...')
        count = 0
        video_file = os.path.join(base_path, filename)
        # Create a VideoCapture object and read from input file
        # If the input is taken from a camera, pass 0 instead of the video file name.
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5)  # Frame rate of the video input.
        while (cap.isOpened()):
            frame_id = cap.get(1)  # Gets current frame number
            ret, frame = cap.read()
            if not ret:
                break
                # Extract all the video frames from the acquired deepfake datasets
            if frame_id % math.floor(frame_rate) == 0:
                print('The original dimensions include: ', frame.shape)
                # In order to cater for different video qualities and to optimize for the image processing performance,
                # the following image resizing strategies were implemented:
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif 1000 < frame.shape[1] <= 1900:
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1
                print('Scale Ratio: ', scale_ratio)

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dimensions = (width, height)
                new_frame = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
                print('Resized Dimensions: ', new_frame.shape)

                new_filename = '{}-{:03d}.png'.format(os.path.join(tmp_path, get_file_name(filename)), count)
                count = count + 1
                cv2.imwrite(new_filename, new_frame)
        cap.release()
        print('Done Converting...!')
    else:
        continue
