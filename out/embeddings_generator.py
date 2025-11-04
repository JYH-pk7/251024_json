import json
import numpy as np
import time
import os
import google.generativeai as genai

# --- 설정 ---
INPUT_JSON_FILE = 'speeches_meeting_50242.json'
EMBEDDING_FILE = 'report_embeddings_google_004.npy' 
METADATA_FILE = 'report_metadata_google_004.json'
MODEL_NAME = 'models/text-embedding-004'     
# -----------

def create_embeddings_index():
    """
    회의록 파일에서 전문위원 보고서만 추출하여
    Google API를 통해 임베딩을 생성하고 파일로 저장합니다.
    """
    
    # 1. API 키 설정
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("오류: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("터미널에서 'export GOOGLE_API_KEY=YOUR_API_KEY'를 실행하세요.")
        return
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"API 키 설정 중 오류 발생: {e}")
        return

    # 2. 데이터 로드
    print(f"'{INPUT_JSON_FILE}' 파일 로드 중...")
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        speeches = json.load(f)

    # 3. '법안 내용 문서' (전문위원 보고) 추출
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

    # 4. 임베딩 생성 (핵심 작업)
    print(f"'{MODEL_NAME}' 모델로 임베딩 생성 중 (API 호출)...")
    start_time = time.time()
    
    report_texts = [report['speech_text'] for report in expert_reports]
    
    try:
        result = genai.embed_content(
            model=MODEL_NAME,
            content=report_texts,
            task_type="RETRIEVAL_DOCUMENT" 
        )
        
        # *** [오류 수정된 부분] ***
        # API 응답인 result['embedding']이 이미 [벡터1, 벡터2, ...] 리스트입니다.
        # 이전 코드: report_embeddings = [r['embedding'] for r in result['embedding']] # <--- 이 부분이 오류의 원인
        report_embeddings = result['embedding'] # <--- 이렇게 수정해야 합니다.
        
        # Numpy 배열로 변환
        report_embeddings_np = np.array(report_embeddings)

    except Exception as e:
        print(f"Google API 호출 중 오류 발생: {e}")
        return
    
    end_time = time.time()
    print(f"임베딩 생성 완료. (소요 시간: {end_time - start_time:.2f}초)")

    # 5. 임베딩 데이터(Numpy 배열) 저장
    try:
        np.save(EMBEDDING_FILE, report_embeddings_np)
        print(f"✅ 임베딩 데이터가 '{EMBEDDING_FILE}' 파일로 저장되었습니다.")
        
        # 6. 메타데이터(JSON) 저장
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(expert_reports, f, ensure_ascii=False, indent=4)
        print(f"✅ 메타데이터가 '{METADATA_FILE}' 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 파일 저장 중 오류 발생: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    create_embeddings_index()