import json
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# --- 설정 ---
INPUT_JSON_FILE = 'speeches_meeting_50242.json'
EMBEDDING_FILE = 'report_embeddings.npy'
METADATA_FILE = 'report_metadata.json'
MODEL_NAME = 'jhgan/ko-sbert-nli'
# -----------

def create_embeddings_index():
    """
    회의록 파일에서 전문위원 보고서만 추출하여
    임베딩(Numpy)과 메타데이터(JSON) 파일로 저장합니다.
    """
    
    # 1. 데이터 로드
    print(f"'{INPUT_JSON_FILE}' 파일 로드 중...")
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        speeches = json.load(f)

    # 2. '법안 내용 문서' (전문위원 보고) 추출
    expert_reports = []
    for speech in speeches:
        member_name = speech.get('member_name', '')
        if '전문위원' in member_name:
            expert_reports.append({
                'speech_id': speech['speech_id'],
                'bills': speech['bills'].replace('\n', ' | '),
                'member_name': member_name,
                'speech_text': speech['speech_text']
            })

    if not expert_reports:
        print("오류: 분석 기준이 될 '전문위원'의 발언을 찾을 수 없습니다.")
        return

    print(f"총 {len(expert_reports)}건의 전문위원 보고서를 찾았습니다.")

    # 3. AI 임베딩 모델 로드
    print(f"'{MODEL_NAME}' 모델 로드 중... (시간이 걸릴 수 있습니다)")
    model = SentenceTransformer(MODEL_NAME)

    # 4. 임베딩 생성 (핵심 작업)
    print("임베딩 생성 중...")
    start_time = time.time()
    
    report_texts = [report['speech_text'] for report in expert_reports]
    report_embeddings = model.encode(report_texts, show_progress_bar=True)
    
    end_time = time.time()
    print(f"임베딩 생성 완료. (소요 시간: {end_time - start_time:.2f}초)")

    # 5. 임베딩 데이터(Numpy 배열) 저장
    try:
        np.save(EMBEDDING_FILE, report_embeddings)
        print(f"✅ 임베딩 데이터가 '{EMBEDDING_FILE}' 파일로 저장되었습니다.")
        
        # 6. 메타데이터(JSON) 저장 (임베딩과 순서가 동일해야 함)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(expert_reports, f, ensure_ascii=False, indent=4)
        print(f"✅ 메타데이터가 '{METADATA_FILE}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 파일 저장 중 오류 발생: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    create_embeddings_index()