import os
import cv2

dst_dir = "/dataset/classification_dataset/train"
src_dir = "/dataset/genki4k/files"

count = 1
for filename in os.listdir(src_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 只处理.jpg和.png文件
        img_path = src_dir + '/' + filename
        img = cv2.imread(img_path)

        if img is not None:  # 检查图片是否成功加载
            if count <= 1200:
                dst_file = dst_dir + '/train/' + 'smile/' + filename
                cv2.imwrite(dst_file, img)
            elif count <= 1700:
                dst_file = dst_dir + '/valid/' + 'smile/' + filename
                cv2.imwrite(dst_file, img)
            elif count <= 2162:
                dst_file = dst_dir + '/test/' + 'smile/' + filename
                cv2.imwrite(dst_file, img)
            elif count <= 3200:
                dst_file = dst_dir + '/train/' + 'non_smile/' + filename
                cv2.imwrite(dst_file, img)
            elif count <= 3700:
                dst_file = dst_dir + '/valid/' + 'non_smile/' + filename
                cv2.imwrite(dst_file, img)
            else:
                dst_file = dst_dir + '/test/' + 'non_smile/' + filename
                cv2.imwrite(dst_file, img)
            print("saving picture ID: " + str(count))
            count += 1