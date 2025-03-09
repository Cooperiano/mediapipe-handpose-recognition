import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from utils import draw_landmarks_on_image


def detect_hands_from_image(img_path):
  # 1、 创建人手坐标点检测器
  # 下载人手关键点检测模型hand_landmarker.task
  # https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  base_options = python.BaseOptions(model_asset_path='PATH_TO_hand_landmarker.task')
  options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
  detector = vision.HandLandmarker.create_from_options(options)

  # 2、 加载输入图片
  image = mp.Image.create_from_file('PATH_TO_image.jpg')

  # 3、 使用下载好的模型进行人手坐标点检测
  detection_result = detector.detect(image)
  print(detection_result)

  # 4、 可视化人手检测
  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  imageRGB = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

  # cv2.imwrite('new_img.jpg', imageRGB)
  # 在使用OpenCV的cv2.imshow函数显示图像时，它会默认将传入的图像数据解释为BGR格式
  # 如果你传入的是RGB格式的图像数据，OpenCV会在显示时进行颜色通道的调整，使图像以BGR格式进行显示。
  cv2.imshow('women_hands', imageRGB)

  # 输入esc结束捕获
  if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_hands_from_image(img_path='PATH_TO_image.jpg')")
