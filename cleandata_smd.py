import os
import numpy as np
import pandas as pd

def convert_txt_to_csv(smd_root, out_root):
    # 三种类型：train/test/test_label → 对应输出：train/test/label
    folder_map = {
        'train': 'train',
        'test': 'test',
        'test_label': 'label'
    }

    for folder_name, out_name in folder_map.items():
        input_path = os.path.join(smd_root, folder_name)
        output_path = os.path.join(out_root, 'SMD', out_name)
        os.makedirs(output_path, exist_ok=True)

        for file in os.listdir(input_path):
            if not file.endswith(".txt"):
                continue

            input_file = os.path.join(input_path, file)
            output_file = os.path.join(output_path, file.replace('.txt', '.csv'))

            try:
                data = np.loadtxt(input_file, delimiter=',')
                df = pd.DataFrame(data)
                df.to_csv(output_file, index=False, header=False)
                print(f"✅ {file} → {out_name}/{os.path.basename(output_file)}")
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

    print("\n🎉 All SMD txt files converted to CSV.")

if __name__ == "__main__":
    # 输入原始 SMD 数据路径（你可以修改成自己的路径）
    smd_root = "./ServerMachineDataset"
    # 输出目录
    out_root = "./clean_data"
    convert_txt_to_csv(smd_root, out_root)
