import json
import os

def main():
    input_file = "제21대 국회 소위원회 법제사법위원회 회의록 데이터셋_speeches.json"  # 원본 파일
    target_meeting_id = 50848     # ✅ 원하는 meeting_id로 변경

    # 1. 출력 파일 이름 자동 생성
    file_name = f"speeches_meeting_{target_meeting_id}.json"
    output_file = os.path.join(os.getcwd(), file_name)

    # 2. JSON 파일 읽기
    with open(input_file, "r", encoding="utf-8") as f:
        speeches = json.load(f)

    # 3. meeting_id로 필터링
    filtered = [s for s in speeches if s.get("meeting_id") == target_meeting_id]

    # 4. 결과 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"✅ meeting_id={target_meeting_id} 에 해당하는 {len(filtered)}개 항목이 '{file_name}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
