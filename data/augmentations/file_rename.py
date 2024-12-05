import os

# 폴더 경로 설정
base_folder = "C:\pycode\dl\project2\project2_3\project2_3\model_test\cctv_abnormal1"

# 모든 파일 경로 출력 및 파일 개수 계산
def list_all_files_with_counts(base_folder):
    total_files = 0
    mp4_files = 0
    xml_files = 0

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            # 파일 전체 경로 생성
            full_path = os.path.join(root, file)
            print(full_path)  # 파일 경로 출력

            # 파일 개수 카운트
            total_files += 1
            if file.endswith(".mp4"):
                mp4_files += 1
            elif file.endswith(".xml"):
                xml_files += 1

    # 결과 출력
    print("\nSummary:")
    print(f"Total files: {total_files}")
    print(f"MP4 files: {mp4_files}")
    print(f"XML files: {xml_files}")


# 함수 실행
list_all_files_with_counts(base_folder)