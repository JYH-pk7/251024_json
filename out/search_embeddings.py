import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- 설정 ---
EMBEDDING_FILE = 'report_embeddings.npy'
METADATA_FILE = 'report_metadata.json'
MODEL_NAME = 'jhgan/ko-sbert-nli'  # 1번 스크립트와 반드시 동일한 모델 사용
TOP_K = 3  # 상위 몇 개까지 표시할지
# -----------

def load_index():
    """ 저장된 임베딩과 메타데이터를 로드합니다. """
    try:
        embeddings = np.load(EMBEDDING_FILE)
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if len(embeddings) != len(metadata):
            print("오류: 임베딩 파일과 메타데이터 파일의 개수가 일치하지 않습니다.")
            return None, None
            
        return embeddings, metadata
    except FileNotFoundError:
        print(f"오류: '{EMBEDDING_FILE}' 또는 '{METADATA_FILE}'을 찾을 수 없습니다.")
        print("먼저 'embeddings_generator.py' 스크립트를 실행해 주세요.")
        return None, None
    except Exception as e:
        print(f"인덱스 로드 중 오류 발생: {e}")
        return None, None

def search_documents(query, model, db_embeddings, db_metadata, top_k):
    """
    검색 쿼리를 임베딩하고 DB와 비교하여 상위 K개 결과를 반환합니다.
    """
    # 1. 검색 쿼리 임베딩 (1개만)
    query_embedding = model.encode([query])
    
    # 2. DB 임베딩과 코사인 유사도 계산
    similarities = cosine_similarity(query_embedding, db_embeddings)
    
    # 3. 유사도 점수가 높은 순서로 인덱스 정렬
    # argsort는 오름차순이므로 [:-top_k-1:-1]을 사용해 상위 K개를 내림차순으로 추출
    top_k_indices = np.argsort(similarities[0])[:-top_k-1:-1]
    
    # 4. 결과 반환
    results = []
    for idx in top_k_indices:
        results.append({
            'score': similarities[0][idx],
            'metadata': db_metadata[idx]
        })
    return results

# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 1. 모델 및 인덱스 로드 (최초 1회)
    print(f"'{MODEL_NAME}' 모델 로드 중...")
    start_time = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"모델 로드 완료. (소요 시간: {time.time() - start_time:.2f}초)")

    print("저장된 임베딩 인덱스 로드 중...")
    db_embeddings, db_metadata = load_index()
    
    if db_embeddings is not None:
        print(f"✅ 총 {len(db_embeddings)}개의 법안 보고서 인덱스 로드 완료.")
        
        # 2. 대화형 검색 루프
        while True:
            try:
                # 사용자 입력 받기
                query = input("\n🔍 검색할 내용을 입력하세요 (종료하려면 'exit' 입력): ")
                if query.lower() == 'exit':
                    print("검색을 종료합니다.")
                    break
                if len(query) < 2:
                    print("검색어가 너무 짧습니다.")
                    continue
                    
                # 3. 검색 수행
                start_time = time.time()
                search_results = search_documents(query, model, db_embeddings, db_metadata, TOP_K)
                end_time = time.time()
                
                print(f"\n--- 검색 결과 (소요 시간: {end_time - start_time:.4f}초) ---")
                
                # 4. 결과 출력
                for i, result in enumerate(search_results):
                    meta = result['metadata']
                    print(f"\n🥇 [유사도 {result['score']:.4f}] - {i+1}위")
                    print(f"   법안: {meta['bills']}")
                    print(f"   보고자: {meta['member_name']} (ID: {meta['speech_id']})")
                    print(f"   내용: {meta['speech_text'][:150]}...")

            except KeyboardInterrupt:
                print("\n검색을 종료합니다.")
                break
            except Exception as e:
                print(f"검색 중 오류 발생: {e}")