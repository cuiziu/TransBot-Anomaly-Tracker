import os


def save_dataset_structure(root_dir, output_file):
    """
    데이터셋 폴더 구조를 txt 파일로 저장하는 함수.

    Args:
        root_dir (str): 데이터셋의 루트 디렉토리 경로.
        output_file (str): 결과를 저장할 txt 파일 경로.
    """

    with open(output_file, 'w') as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 현재 디렉토리 경로 기록
            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level  # 계층에 따라 들여쓰기
            f.write(f"{indent}[{os.path.basename(dirpath)}]\n")

            # 파일 이름 기록
            subindent = ' ' * 4 * (level + 1)
            for filename in filenames:
                f.write(f"{subindent}{filename}\n")

    print(f"폴더 구조가 '{output_file}'에 저장되었습니다.")


# 사용 예시
dataset_root = 'C:\pycode\dl\project2\project2_3\project2_3\model_test\cctv_abnormal\insidedoor_03'  # 데이터셋 루트 폴더 경로
output_txt_file = 'dataset_structure.txt'  # 저장할 txt 파일 이름
save_dataset_structure(dataset_root, output_txt_file)
