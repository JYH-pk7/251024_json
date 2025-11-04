import os
import json
import re
import time
from openai import OpenAI
import logging

# --- 설정 ---

# 로깅 설정 (진행 상황 및 오류 확인용)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. OpenAI 클라이언트 초기화 (환경 변수에서 API 키 로드)
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    logging.error("터미널에서 'export OPENAI_API_KEY=YOUR_KEY'를 실행하세요.")
    exit()
# output file은 입력 파일과 동일한 폴더에 저장됩니다.
INPUT_FILE = 'speeches_meeting_50848.json'
# 입력 파일명에서 회의 코드를 추출하여 출력 파일명 생성
meeting_code = INPUT_FILE.split('_')[2].split('.')[0]  # '50242' 추출
OUTPUT_FILE = f'processed_speeches_{meeting_code}.json'

# 사용할 LLM 모델 (gpt-4o-mini는 빠르고 비용 효율적입니다)
LLM_MODEL = "gpt-4o-mini" 

# --- 헬퍼 함수 ---

def get_bill_name(bill_string):
    """
    법안 전체 문자열에서 법안의 고유한 '이름' 부분만 추출합니다.
    예: "1. 교정공제회법 일부개정법률안(박홍근 의원 대표발의)" -> "교정공제회법 일부개정법률안"
    """
    # Regex: 시작(^) -> 숫자(\d+) -> 점(\.) -> 공백(\s*) -> [캡처 시작](.*?) -> 첫 괄호 '()' 또는 문자열 끝($)
    match = re.search(r'^\d+\.\s*(.*?)(?:\(|$)', bill_string.strip())
    if match:
        return match.group(1).strip()
    
    # "개의" 같이 번호가 없는 경우 원본 문자열 반환
    return bill_string.strip()

def find_most_relevant_bill(speech_text, bill_list):
    """
    LLM을 사용하여 발언 내용과 가장 일치하는 법안 1개를 찾습니다.
    """
    
    # LLM에게 선택지를 명확하게 주기 위해 법안 목록을 문자열로 변환
    bill_options_str = "\n".join(f"- {bill}" for bill in bill_list)

    # ChatGPT에 보낼 프롬프트
    prompt = f"""
    당신은 국회 회의록을 정확하게 분석하는 AI 어시스턴트입니다.
    아래 제공되는 [발언 내용]을 분석하여, [법안 목록] 중에서 가장 밀접하게 관련된 법안 **단 하나**만 선택해 주십시오.

    [발언 내용]
    {speech_text}

    [법안 목록]
    {bill_options_str}

    [요청]
    [법안 목록]에 있는 법안의 **전체 텍스트**를 그대로 복사하여 응답해 주십시오.
    (예: '3. 출입국관리법 일부개정법률안(백혜련 의원 대표발의)(의안번호 1663)')
    
    **어떠한 설명이나 추가 텍스트도 포함하지 말고, 오직 선택한 법안의 전체 텍스트만 응답해야 합니다.**
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "당신은 국회 회의록을 분석하는 정확한 AI 어시스턴트입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # 일관성 있는 답변을 위해 0.0으로 설정
        )
        
        chosen_bill = response.choices[0].message.content.strip()

        # LLM 응답이 원본 목록에 있는지 확인 (가장 정확한 매칭)
        for original_bill in bill_list:
            # LLM이 미세하게 다르게 응답할 경우를 대비
            if chosen_bill == original_bill or chosen_bill in original_bill or original_bill in chosen_bill:
                 return original_bill # 일관성을 위해 원본 리스트의 문자열 반환

        # LLM이 목록에 없는 이상한 응답을 한 경우
        logging.warning(f"LLM 응답 '{chosen_bill}'을(를) 원본 법안 목록에서 찾을 수 없습니다. 첫 번째 법안으로 대체합니다.")
        return bill_list[0] # 실패 시, 첫 번째 법안을 기본값으로 반환

    except Exception as e:
        logging.error(f"OpenAI API 호출 중 오류 발생: {e}")
        return None # 오류 발생 시 None 반환

# --- 메인 처리 로직 ---

def process_speeches():
    """
    JSON 파일을 읽고, 'bills' 필드를 로직에 따라 처리한 후 새 파일로 저장합니다.
    """
    logging.info(f"{INPUT_FILE} 파일을 불러오는 중...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            speeches = json.load(f)
    except FileNotFoundError:
        logging.error(f"입력 파일을 찾을 수 없습니다: {INPUT_FILE}")
        return
    except json.JSONDecodeError:
        logging.error(f"{INPUT_FILE} 파일의 JSON 형식이 올바르지 않습니다.")
        return

    processed_speeches = []
    total_speeches = len(speeches)

    for i, speech in enumerate(speeches):
        # 원본 데이터를 수정하지 않기 위해 사본 생성
        new_speech = speech.copy()
        
        bill_string = speech.get('bills', '')
        speech_text = speech.get('speech_text', '')

        # 'bills' 필드를 '\n' 기준으로 분리하여 리스트 생성
        bill_list = [b.strip() for b in bill_string.split('\n') if b.strip()]

        # --- 요청하신 로직 수행 ---

        # 1. 법안이 1개 이하인 경우 (0개 또는 1개)
        if len(bill_list) <= 1:
            processed_speeches.append(new_speech)
            continue
        
        # 2. 법안이 2개 이상인 경우
        logging.info(f"[{i+1}/{total_speeches}] ID: {speech['speech_id']} - 법안 {len(bill_list)}개 발견, 분석 시작...")
        
        # 2-1. 법안 이름 추출 및 비교
        bill_names = [get_bill_name(b) for b in bill_list]
        
        # set을 사용해 고유한 이름이 1개인지 확인
        if len(set(bill_names)) == 1:
            # Case 2a: 이름이 모두 동일하면, 원본 'bills' 필드 유지
            logging.info(f"  -> 법안 이름이 모두 동일합니다. 원본 유지.")
            processed_speeches.append(new_speech)
        else:
            # Case 2b: 이름이 하나라도 다르면, LLM 분석 수행
            logging.warning(f"  -> 법안 이름이 다릅니다. LLM 분석을 수행합니다...")
            
            most_relevant_bill = find_most_relevant_bill(speech_text, bill_list)
            
            if most_relevant_bill:
                logging.info(f"  -> LLM 선택: {most_relevant_bill}")
                new_speech['bills'] = most_relevant_bill # 'bills' 필드를 1개의 법안으로 교체
            else:
                logging.warning(f"  -> LLM 분석 실패. ID: {speech['speech_id']}. 원본 'bills' 필드 유지.")
                # new_speech는 이미 원본 speech의 복사본이므로 별도 처리 필요 없음
            
            processed_speeches.append(new_speech)

    # --- 결과 저장 ---
    logging.info(f"모든 처리 완료. {OUTPUT_FILE} 파일로 저장 중...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_speeches, f, ensure_ascii=False, indent=4)
        logging.info(f"성공적으로 {OUTPUT_FILE} 파일에 저장했습니다.")
    except IOError as e:
        logging.error(f"결과 파일 저장 중 오류 발생: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    import time
    start_time = time.time()  # 시작 시간 기록
    
    process_speeches()
    
    end_time = time.time()  # 종료 시간 기록
    execution_time = end_time - start_time
    print(f"\n실행 시간: {execution_time:.2f}초")