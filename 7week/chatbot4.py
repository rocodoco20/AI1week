import streamlit as st
import base64
import openai
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 환경 변수 로드
load_dotenv("/home/ubuntu/hanghaeAI/API_KEY.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API 키가 설정되지 않았습니다. 'API_KEY.env' 파일을 확인하세요.")
    st.stop()

# OpenAI API 초기화
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

# Streamlit UI 설정
st.title("소비 패턴 기반 패션 추천 서비스")
st.write("인물 사진과 물건 사진을 업로드하고, 패션 추천을 받아보세요!")

# 인물 사진 업로드
st.write("## 인물 사진 업로드")
person_files = st.file_uploader(
    "인물 사진을 업로드하세요 (최대 2장)", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)
person_image_data = []

if person_files:
    st.write(f"업로드된 인물 사진: {len(person_files)}장")
    for img in person_files:
        st.image(img, caption="업로드된 인물 사진")
        # 이미지 데이터를 Base64로 인코딩
        image_base64 = base64.b64encode(img.read()).decode("utf-8")
        person_image_data.append(image_base64)

# 물건 사진 업로드
st.write("## 물건 사진 업로드")
item_files = st.file_uploader(
    "물건 사진을 업로드하세요 (최대 3장)", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)
item_image_data = []

if item_files:
    st.write(f"업로드된 물건 사진: {len(item_files)}장")
    for img in item_files:
        st.image(img, caption="업로드된 물건 사진")
        # 이미지 데이터를 Base64로 인코딩
        image_base64 = base64.b64encode(img.read()).decode("utf-8")
        item_image_data.append(image_base64)

# 질의응답 섹션
if person_image_data or item_image_data:
    st.write("### 질문을 입력하세요")
    question = st.text_input("패션 추천을 위해 원하는 스타일 또는 상황을 입력하세요:")

    if question:
        # 모델에 전달할 메시지 구성
        prompt_content = [
            {"type": "text", "text": f"사용자가 업로드한 인물 사진 {len(person_image_data)}장과 물건 사진 {len(item_image_data)}장을 기반으로 다음 질문에 답하세요: {question}"}
        ]
        
        # 인물 사진 추가
        for img_base64 in person_image_data:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })

        # 물건 사진 추가
        for img_base64 in item_image_data:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })

        # LangChain HumanMessage 생성
        message = HumanMessage(prompt_content)

        # 모델 호출 및 응답 처리
        with st.spinner("추천 결과를 분석 중입니다..."):
            try:
                result = model.invoke([message])  # 모델 실행
                response = result.content
                st.success("패션 추천 결과:")
                st.markdown(response)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

# 참고 정보
st.write("---")
st.info("이미지는 개인 정보 보호를 위해 안전하게 처리됩니다.")
