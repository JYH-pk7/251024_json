import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def analyze_speech_similarity(json_file_path, output_json_path='analysis_results.json'):
    """
    회의록 JSON 파일을 읽어, 전문위원 보고 내용과
    다른 의원들의 발언 간의 의미 유사도를 분석하고 결과를 JSON 파일로 저장합니다.
    """
    
    # 1. 데이터 로드
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            speeches = json.load(f)
    except FileNotFoundError:
        print(f"오류: '{json_file_path}' 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        return

    # 2. AI 임베딩 모델 로드 (한국어 특화 모델)
    try:
        model = SentenceTransformer('jhgan/ko-sbert-nli')
        print("AI 임베딩 모델(ko-sbert-nli) 로드 중...")
    except Exception as e:
        print(f"ko-sbert-nli 모델 로드 실패. 대체 모델(다국어)을 사용합니다. (오류: {e})")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 3. '법안 내용 문서' (전문위원 보고) 추출
    expert_reports = []
    for speech in speeches:
        member_name = speech.get('member_name', '')
        if '전문위원' in member_name:
            expert_reports.append({
                'speech_id': speech['speech_id'],
                'bills_key': speech['bills'],
                'description': speech['speech_text']
            })

    if not expert_reports:
        print("분석 기준이 될 '전문위원'의 발언을 찾을 수 없습니다.")
        return

    # 4. '법안 내용 문서'에 대한 임베딩 미리 계산
    print(f"전문위원 보고서 {len(expert_reports)}건의 임베딩을 생성합니다...")
    report_descriptions = [report['description'] for report in expert_reports]
    report_embeddings = model.encode(report_descriptions)

    # 5. '의원 발언' (쿼리)을 분석하여 가장 유사한 '법안 내용' 매칭
    print("의원 발언을 분석하고 유사도를 계산합니다...")
    analysis_results = []
    total_speeches = len(speeches)
    
    for i, speech in enumerate(speeches):
        member_name = speech.get('member_name', '')
        speech_text = speech['speech_text']

        # 전문위원 본인 발언 및 너무 짧은 발언(예: "예.")은 분석에서 제외
        if '전문위원' in member_name or len(speech_text) < 30:
            continue
            
        # 진행 상태 표시 (선택 사항)
        # if (i + 1) % 100 == 0:
        #     print(f"  진행 중: {i+1} / {total_speeches} 발언 처리 중...")

        # 현재 발언(쿼리)의 임베딩 생성
        speech_embedding = model.encode([speech_text])

        # 모든 '법안 내용 문서'와의 유사도 계산
        similarities = cosine_similarity(speech_embedding, report_embeddings)

        # 가장 점수가 높은 매칭 항목 찾기
        best_match_index = np.argmax(similarities)
        best_score = similarities[0][best_match_index]
        matched_report = expert_reports[best_match_index]

        analysis_results.append({
            'speech_id': speech['speech_id'],
            'member_name': member_name,
            'speech_text': speech_text, # 스니펫 대신 전체 텍스트 저장
            'score': float(best_score),
            'matched_report_id': matched_report['speech_id'],
            'matched_bills': matched_report['bills_key'].replace('\n', ' | ')
        })

    # 6. 결과를 유사도 점수 순으로 정렬
    analysis_results.sort(key=lambda x: x['score'], reverse=True)
    
    # 7. *** [추가된 코드] ***
    #    전체 분석 결과를 JSON 파일로 저장
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 분석 결과 총 {len(analysis_results)}건을 '{output_json_path}' 파일로 성공적으로 저장했습니다.")
    except Exception as e:
        print(f"\n❌ 분석 결과를 파일로 저장하는 데 실패했습니다: {e}")


    # 8. 콘솔에 상위 결과 일부 출력
    print("\n--- 분석 결과 (유사도 상위 10건) ---")
    print("-" * 70)

    for result in analysis_results[:10]:
        print(f"🗣️  발언 ID: {result['speech_id']} (발언자: {result['member_name']})")
        print(f"   내용: \"{result['speech_text'][:80]}...\"")
        print(f"   ➡️  매칭 법안 (유사도: {result['score']:.4f}):")
        print(f"       {result['matched_bills']} (기준 보고서 ID: {result['matched_report_id']})")
        print("-" * 70)


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 스크립트와 동일한 위치에 'speeches_meeting_50242.json' 파일이 있다고 가정
    # 저장될 파일명은 'analysis_results.json' 입니다.
    analyze_speech_similarity(
        json_file_path='speeches_meeting_50242.json',
        output_json_path='analysis_results.json'
    )