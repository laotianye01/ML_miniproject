import os
import cv2

dst_dir = "/dataset/3D_dataset"
src_dir = "/dataset/genki4k/files"
txt_path = "/dataset/genki4k/labels.txt"

train_txt = "ml_miniproject/dataset/3D_dataset/train.txt"
val_txt = "ml_miniproject/dataset/3D_dataset/valid.txt"

count = 0
for filename in os.listdir(src_dir):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    tri = open(train_txt, 'a')
    val = open(val_txt, 'a')
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 只处理.jpg和.png文件
        img_path = src_dir + '/' + filename
        img = cv2.imread(img_path)

        if img is not None:  # 检查图片是否成功加载
            if count <= 1500:
                dst_file = dst_dir + '/train/' + filename
                tri.writelines(lines[count])
                cv2.imwrite(dst_file, img)
            elif count <= 2162:
                dst_file = dst_dir + '/valid/' + filename
                val.writelines(lines[count])
                cv2.imwrite(dst_file, img)
            elif count <= 3450:
                dst_file = dst_dir + '/train/' + filename
                tri.writelines(lines[count])
                cv2.imwrite(dst_file, img)
            else:
                dst_file = dst_dir + '/valid/' + filename
                val.writelines(lines[count])
                cv2.imwrite(dst_file, img)
            print("saving picture ID: " + str(count))
            count += 1