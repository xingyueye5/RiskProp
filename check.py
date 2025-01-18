import os


def check_images_continuity(root_folder):
    """
    检查给定文件夹中所有名为 'images' 的子文件夹中是否都是从 000001.jpg 开始连续的图像文件。
    """
    all_continuous = True

    for dirpath, dirnames, filenames in os.walk(root_folder):
        if "images" in dirnames:
            continuous = True
            if dirpath.split("/")[-1] in ["004928", "007155"]:
                continue
            images_folder = os.path.join(dirpath, "images")
            image_files = sorted(f for f in os.listdir(images_folder) if f.endswith(".jpg"))

            expected_index = 1  # 从 000001 开始
            for filename in image_files:
                file_index = int(filename.split(".")[0])
                if file_index != expected_index:
                    print(
                        f"Discontinuity found in {images_folder}: Expected {expected_index:06d}.jpg, found {filename}"
                    )
                    continuous = False
                    all_continuous = False
                    break
                expected_index += 1

            if continuous:
                pass
                # print(f"Folder {images_folder} is continuous.")
            else:
                print(f"Folder {images_folder} is NOT continuous.")

    if all_continuous:
        print("All 'images' folders have continuous image files.")
    else:
        print("Some 'images' folders have discontinuous image files.")


# 使用方式
root_folder = "data/MM-AU/CAP-DATA"
check_images_continuity(root_folder)
