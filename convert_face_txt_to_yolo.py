import os

def parse_wider_txt(txt_path, image_dir, label_dir, image_size=(640, 480)):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        filename = lines[idx].strip()
        idx += 1
        if idx >= len(lines): break  

        try:
            face_count = int(lines[idx].strip())
        except ValueError:
            print(f"[ERROR] cant pase number of faces, position：{idx}, content：{lines[idx]}")
            break

        idx += 1

        label_lines = []
        for _ in range(face_count):
            if idx >= len(lines): break 
            parts = list(map(int, lines[idx].strip().split()))
            x, y, w, h = parts[:4]
            x_center = (x + w / 2) / image_size[0]
            y_center = (y + h / 2) / image_size[1]
            w_norm = w / image_size[0]
            h_norm = h / image_size[1]
            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            idx += 1

        base_name = os.path.splitext(os.path.basename(filename))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")
        with open(label_path, 'w') as lf:
            lf.write('\n'.join(label_lines))



if __name__ == "__main__":
    os.makedirs("Dataset/Face/WIDER_train/labels", exist_ok=True)
    os.makedirs("Dataset/Face/WIDER_val/labels", exist_ok=True)

    parse_wider_txt("Dataset/Face/labels/wider_face_train_bbx_gt.txt",
                "Dataset/Face/WIDER_train/images",
                "Dataset/Face/WIDER_train/labels")

    parse_wider_txt("Dataset/Face/labels/wider_face_val_bbx_gt.txt",
                "Dataset/Face/WIDER_val/images",
                "Dataset/Face/WIDER_val/labels")


    print("WIDER FACE annotations converted to YOLO format.")
