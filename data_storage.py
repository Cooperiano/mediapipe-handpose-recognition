import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import csv
from utils import draw_landmarks_on_image

# 将指定路径的图片文件夹中的所有图片进行人手关键点检测，并将结果保存到 CSV 文件中
# 设置模型路径
MODEL_PATH = 'PATH_TO_hand_landmarker.task'

# 设置图片文件夹路径
IMAGE_FOLDER = 'PATH_TO_IMAGES'

# 结果存储路径（CSV 文件）
OUTPUT_CSV = 'PATH_TO_hand_landmarks.csv'
def detect_hands_from_images(image_folder, output_csv):
    # 1. 初始化 MediaPipe 人手检测器
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # 2. 创建 CSV 文件并写入表头
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "hand_index", "landmark_index", "x", "y", "z"])  # 表头

        # 3. 遍历所有图片
        for img_file in os.listdir(image_folder):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):  # 只处理图片
                img_path = os.path.join(image_folder, img_file)
                print(f"Processing: {img_path}")

                # 4. 读取图片并进行检测
                image = mp.Image.create_from_file(img_path)
                detection_result = detector.detect(image)
                hand_landmarks = detection_result.hand_landmarks

                # 5. 解析关键点信息
                for hand_index, hand in enumerate(hand_landmarks):  # 多只手
                    for lm_index, lm in enumerate(hand):  # 每只手有 21 个关键点
                        writer.writerow([img_file, hand_index, lm_index, lm.x, lm.y, lm.z])

                # 6. 可视化（可选）
                annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
                imageRGB = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                cv2.imshow('Hand Detection', imageRGB)

                # 按 ESC 退出
                if cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    break

    print(f"所有手部关键点数据已保存到 {output_csv}")

if __name__ == '__main__':
    detect_hands_from_images(IMAGE_FOLDER, OUTPUT_CSV)
