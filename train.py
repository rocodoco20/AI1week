import os
import sys
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers
import numpy as np  # 추가된 부분

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
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

    dataset_name: Optional[str] = field(default='wikitext')  # 사용할 데이터셋 이름 (wikitext로 변경)
    dataset_config_name: Optional[str] = field(default='plain_text')  # 'plain_text'로 설정 (유효한 데이터셋 설정)
    block_size: int = field(default=1024)  # 입력 텍스트의 길이
    num_workers: Optional[int] = field(default=2)  # 데이터 로딩 시 사용할 워커 수
    custom_output_dir: str = field(default='./output')  # 변경된 output_dir 변수명

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

# 데이터셋 로드
raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

# 훈련 데이터 일부만 사용
train_dataset = raw_datasets["train"].select(range(10000))

# text 컬럼에서 결측값 및 NaN 값 처리
column_names = list(train_dataset.features)
text_column_name = "text" if "text" in column_names else column_names[0]

# 결측값 및 NaN 값 필터링
train_dataset = train_dataset.filter(lambda example: example[text_column_name] is not None and example[text_column_name] != "")
train_dataset = train_dataset.filter(lambda example: example[text_column_name] != np.nan)

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
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

# 모델의 임베딩 사이즈 조정
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# 텍스트 토크나이즈 함수
def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    output["input_ids"] = [list(map(int, ids)) for ids in output["input_ids"]]  # 타입 캐스팅
    return output

# 토크나이즈 처리
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,  # num_proc을 1로 설정하여 멀티프로세싱을 방지
        remove_columns=column_names
    )

# 최대 포지션 임베딩 크기 설정
max_pos_embeddings = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

# 텍스트를 블록 사이즈에 맞게 그룹화
def group_texts(examples):
    # example이 여러 개의 input_ids를 가진 형태로 변환됩니다.
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

train_dataset = lm_datasets

# DataCollatorForLanguageModeling 사용하여 tokenizer 대체
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # CausalLM 모델이므로, 마스크드 언어 모델링 사용 안함
)

# compute_metrics 함수 추가 (eval_loss 계산용)
def compute_metrics(p):
    eval_loss = p.loss  # Trainer에서 직접 제공하는 손실
    return {"eval_loss": eval_loss}

# TrainingArguments에서 배치 크기 지정
training_args.per_device_train_batch_size = 2  # 배치 크기 2로 설정

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# 체크포인트 확인 및 학습 시작
checkpoint = None
last_checkpoint = get_last_checkpoint(output_dir)  # 변경된 output_dir 사용
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

# WandB 로그 기록 (train_loss와 eval_loss 모두 기록)
wandb.log({
    "train_loss": metrics['train_loss'],
    "eval_loss": metrics['eval_loss']
})  # train_loss와 eval_loss를 각각 WandB에 기록