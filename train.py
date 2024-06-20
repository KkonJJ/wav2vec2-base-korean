# 필요한 라이브러리 임포트
from datasets import load_dataset, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import re
import json
import torch
import numpy as np
from datasets import load_metric
from huggingface_hub import notebook_login
from tqdm import tqdm

# Hugging Face Hub 로그인
notebook_login()

# TensorBoard 확장 로드 및 서버 시작
%load_ext tensorboard
%tensorboard --logdir ./logs

# 데이터셋 로드 및 전처리
chars_to_ignore_regex = '[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    return batch

# 데이터셋 로드
print("Loading dataset...")
zeroth_data = load_dataset("kresnik/zeroth_korean", "clean")

# 데이터셋에서 불필요한 컬럼 제거
print("Removing unnecessary columns...")
zeroth_data = zeroth_data.remove_columns(['speaker_id', 'chapter_id', 'id'])

# 특수 문자 제거
print("Removing special characters from text...")
zeroth_data = zeroth_data.map(remove_special_characters)

# Vocab 파일 생성 및 토크나이저 준비
vocab_dict = {' ': 0, 'ㄱ': 1, 'ㄲ': 2, 'ㄴ': 3, 'ㄷ': 4, 'ㄸ': 5, 'ㄹ': 6, 'ㅁ': 7, 'ㅂ': 8, 'ㅃ': 9, 'ㅅ': 10, 'ㅆ': 11, 'ㅇ': 12, 'ㅈ': 13, 'ㅉ': 14, 'ㅊ': 15, 'ㅋ': 16, 'ㅌ': 17, 'ㅍ': 18, 'ㅎ': 19, 'ㅏ': 20, 'ㅐ': 21, 'ㅑ': 22, 'ㅒ': 23, 'ㅓ': 24, 'ㅔ': 25, 'ㅕ': 26, 'ㅖ': 27, 'ㅗ': 28, 'ㅛ': 29, 'ㅜ': 30, 'ㅠ': 31, 'ㅡ': 32, 'ㅣ': 33, '[UNK]': 34, '[PAD]': 35}
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

print("Loading tokenizer...")
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# Feature Extractor와 Processor 준비
print("Loading feature extractor and processor...")
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# 데이터셋 준비 함수
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

print("Preparing dataset...")
zeroth_data = zeroth_data.map(prepare_dataset, remove_columns=zeroth_data.column_names["train"], num_proc=12)

# 모델 로드 및 설정
print("Loading model...")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.to("cuda")

# gradient_checkpointing_enable() 사용
model.gradient_checkpointing_enable()

# DataCollatorCTCWithPadding 정의
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

# Trainer 설정
training_args = TrainingArguments(
  output_dir="./wav2vec2-base-korean",
  group_by_length=True,
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  warmup_steps=500,
  save_total_limit=2,
  logging_dir='./logs',  # TensorBoard 로그 디렉토리
  report_to="tensorboard",  # TensorBoard로 로그를 보고
)

# Data collator와 metrics 설정
print("Loading data collator and metric...")
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
cer_metric = load_metric("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# Trainer 생성 및 훈련
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=zeroth_data["train"],
    eval_dataset=zeroth_data["test"],
    tokenizer=processor.feature_extractor,
)

print("Training model...")
trainer.train()

# 모델 평가
def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    return batch

print("Evaluating model...")
results = zeroth_data["test"].map(map_to_result, remove_columns=zeroth_data["test"].column_names)
print("Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], references=results["text"])))

# 최종 모델과 프로세서 저장
print("Saving model and processor...")
model.push_to_hub("wav2vec2-base-korean")
processor.push_to_hub("wav2vec2-base-korean")
