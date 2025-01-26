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

st.title("Fashion Recommendation Bot")
if image := st.file_uploader("본인의 전신이 보이는 사진을 올려주세요!", type=['png', 'jpg', 'jpeg']):
    st.image(image)
    image = base64.b64encode(image.read()).decode("utf-8")
    with st.chat_message("assistant"):
        message = HumanMessage(
            content=[
                {"type": "text", "text": "사람의 전신이 찍혀있는 사진이 한 장 주어집니다. 이 때, 사진 속 사람과 어울리는 옷 및 패션 스타일을 추천해주세요."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ],
        )
        result = model.invoke([message])
        response = result.content
        st.markdown(response)