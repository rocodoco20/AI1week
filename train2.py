import os
import sys
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers
import numpy as np

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint

# Weights and Biases 초기화
wandb.init(project='Hanghae99')
wandb.run.name = 'gpt-finetuning'

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default='gpt2')  # 사용할 모델 이름
    torch_dtype: Optional[str] = field(default='auto', metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})

    dataset_name: Optional[str] = field(default=None)  # 사용자 정의 데이터셋 사용
    block_size: int = field(default=1024)  # 입력 텍스트의 길이
    num_workers: Optional[int] = field(default=2)  # 데이터 로딩 시 사용할 워커 수
    custom_output_dir: str = field(default='./output')  # 변경된 output_dir 변수명
    corpus_path: str = field(default='./hanghaeAI/corpus.json')  # corpus.json 경로

# 명령어 인자 파싱
parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# output_dir을 수정하여 실제 경로를 사용
output_dir = args.custom_output_dir if args.custom_output_dir != '/path' else './output'  # 예시로 './output'을 사용

# output_dir 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 디렉토리 권한 부여
os.chmod(output_dir, 0o777)  # 모든 사용자에게 쓰기 권한 부여

# 로깅 설정
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.info(f"Training/evaluation parameters {training_args}")

# 사용자 정의 corpus.json 데이터셋 로드
import json

# corpus.json을 불러와서 Dataset 형식으로 변환
with open(args.corpus_path, "r") as f:
    corpus_data = json.load(f)

# "input"과 "output"을 텍스트로 연결하여 단일 텍스트 시퀀스로 구성
train_data = [{"text": f"Instruction: {item['input']} Response: {item['output']}"} for item in corpus_data]

# Dataset으로 변환
train_dataset = Dataset.from_list(train_data)

# 텍스트 컬럼에서 결측값 및 NaN 값 처리
column_names = list(train_dataset.features)
text_column_name = "text" if "text" in column_names else column_names[0]

# 결측값 및 NaN 값 필터링
train_dataset = train_dataset.filter(lambda example: example[text_column_name] is not None and example[text_column_name] != "")
train_dataset = train_dataset.filter(lambda example: example[text_column_name] != np.nan)

# 훈련 데이터셋과 평가 데이터셋 분리
split_dataset = train_dataset.train_test_split(test_size=0.1)  # 10%를 평가 데이터셋으로 분리
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 모델과 토크나이저 설정
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# 패딩 토큰 설정
tokenizer.pad_token_id = tokenizer.eos_token_id

# 모델의 임베딩 사이즈 조정
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    output = tokenizer(examples[text_column_name], truncation=True, padding="max_length", max_length=args.block_size)
    print(output)  # 이 부분을 추가해서 반환된 값을 출력해보세요
    return output

# 토크나이즈 처리
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,  # num_proc을 1로 설정하여 멀티프로세싱을 방지
        remove_columns=[text_column_name]  # 'text' 컬럼 제거
    )

# 최대 포지션 임베딩 크기 설정
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

# 텍스트를 블록 사이즈에 맞게 그룹화: 'input_ids' 기반
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()  # "labels"을 "input_ids"와 동일하게 설정
    return result


# 그룹화 처리
with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1  # num_proc을 1로 설정하여 멀티프로세싱을 방지
    )

train_dataset = lm_datasets  # 학습 데이터셋으로 변환

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # CausalLM 모델이므로, 마스크드 언어 모델링 사용 안함
)

# compute_metrics 함수 추가 (eval_loss 계산용)
def compute_metrics(p):
    eval_loss = p.loss  # eval_loss는 Trainer에서 자동으로 계산됨
    return {"eval_loss": eval_loss}

# TrainingArguments에서 배치 크기 지정
training_args.per_device_train_batch_size = 2  # 배치 크기 2로 설정
training_args.evaluation_strategy = "epoch"  # 매 epoch마다 평가 실행
# TrainingArguments에서 remove_unused_columns=False 추가
training_args.remove_unused_columns = False  # 'text' 컬럼을 제거하지 않도록 설정

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # eval_dataset 추가
    data_collator=data_collator,
    compute_metrics=compute_metrics  # compute_metrics 함수 추가
)

# 체크포인트 확인 및 학습 시작
checkpoint = None
last_checkpoint = get_last_checkpoint(output_dir)
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
else:
    checkpoint = last_checkpoint

train_result = trainer.train(resume_from_checkpoint=checkpoint)

# 모델 저장
trainer.save_model()

# 메트릭스 저장
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# 평가 실행 (eval_loss 포함)
eval_result = trainer.evaluate()
print(f"Evaluation loss: {eval_result['eval_loss']}")

# WandB 로그 기록 (train_loss와 eval_loss 모두 기록)
wandb.log({
    "train_loss": metrics['train_loss'],
    "eval_loss": eval_result['eval_loss']  # 평가 손실 기록
})
