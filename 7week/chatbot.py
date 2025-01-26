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
# LangChain의 ChatOpenAI를 사용할 때 API 키를 직접 전달해야 합니다
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

# Streamlit UI 설정
st.title("다중 이미지 기반 Q&A 챗봇")
st.write("여러 이미지를 업로드하고 질문을 통해 분석 결과를 받아보세요.")

# 이미지 업로드
uploaded_files = st.file_uploader("이미지를 업로드하세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
image_data = []

if uploaded_files:
    st.write(f"업로드된 이미지: {len(uploaded_files)}장")
    for img in uploaded_files:
        st.image(img)
        # 이미지 데이터를 Base64로 인코딩
        image_base64 = base64.b64encode(img.read()).decode("utf-8")
        image_data.append(image_base64)

# 질의응답
if image_data:
    question = st.text_input("질문을 입력하세요:")
    if question:
        # 이미지 출력
        for img in uploaded_files:
            st.image(img)
        
        # 모델에 전달할 메시지 구성
        with st.chat_message("assistant"):
            message_content = []
            message_content = [{
                    "type": "text",
                    "text": f"사용자가 업로드한 {len(image_data)}장의 이미지를 분석하고, 다음 질문에 답하세요: {question}"
                }]
            for img_base64 in image_data:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                })
            message = HumanMessage(
                message_content
            )

            result = model.invoke([message])  # 모델 실행
            response = result.content
            st.markdown(response)
