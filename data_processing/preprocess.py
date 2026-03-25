import os
import cv2

def crop_and_align_face(image_path, output_path, margin=1.3, output_size=(256, 256)):
    """
    参考 DeepfakeBench 的数据预处理流程:
    包含人脸检测、边缘扩展(Margin)和统一缩放为256x256
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 2. 使用 OpenCV 自带的 Haar 级联分类器进行基础人脸检测 (小白友好，无需复杂配置)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(f"未检测到人脸: {image_path}")
        return

    # 3. 提取第一张人脸的坐标 (x, y, w, h)
    x, y, w, h = faces[0]

    # 4. 根据 DeepfakeBench 规范，应用 Margin (默认1.3) 来包含更多上下文
    # Margin 过大容易导致模型过拟合背景，过小则丢失轮廓信息
    center_x, center_y = x + w // 2, y + h // 2
    new_w = int(w * margin)
    new_h = int(h * margin)
    
    # 计算新的边界框（防止越界）
    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(img.shape[1], center_x + new_w // 2)
    y2 = min(img.shape[0], center_y + new_h // 2)

    # 5. 裁剪人脸
    cropped_face = img[y1:y2, x1:x2]

    # 6. 统一缩放到 256x256
    aligned_face = cv2.resize(cropped_face, output_size)

    # 7. 保存处理后的人脸
    cv2.imwrite(output_path, aligned_face)
    print(f"成功处理并保存: {output_path}")

if __name__ == "__main__":
    # 这里是测试代码：你可以自己在项目里建个 raw_data 文件夹放张照片试试
    print("DeepfakeBench 数据预处理模块已加载...")
    print("目前采用 OpenCV 基础人脸检测，后续可升级为 RetinaFace 或 Dlib 获取关键点 (Landmarks)。")
    
    # 建立一个输出文件夹用来存结果
    os.makedirs("../dataset/processed_faces", exist_ok=True)
    # 假设你有一张测试图叫 test.jpg，取消下面两行的注释就可以跑通
    # input_img = "test.jpg"  
    # crop_and_align_face(input_img, "../dataset/processed_faces/test_aligned.jpg")