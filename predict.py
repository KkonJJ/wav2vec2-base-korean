{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Wav2Vec2 모델을 사용한 한국어 음성 인식\n",
    "이 노트북에서는 Hugging Face의 Wav2Vec2 모델을 사용하여 한국어 음성을 텍스트로 변환합니다."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 필요한 라이브러리 임포트\n",
    "import torch\n",
    "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor\n",
    "import torchaudio\n",
    "from jamo import h2j, j2hcj, is_jamo"
   ],
   "execution_count": 1,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 모델 및 프로세서 로드\n",
    "사전 훈련된 Wav2Vec2 모델과 프로세서를 Hugging Face Hub에서 로드합니다."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 모델과 프로세서 로드\n",
    "model_name = \"Kkonjeong/wav2vec2-base-korean\"\n",
    "processor = Wav2Vec2Processor.from_pretrained(model_name)\n",
    "model = Wav2Vec2ForCTC.from_pretrained(model_name)\n",
    "\n",
    "# 모델을 평가 모드로 설정\n",
    "model.eval()\n",
    "model.to(\"cuda\")"
   ],
   "execution_count": 2,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 자모 결합 함수\n",
    "자모(Jamo)를 결합하여 완전한 한글 텍스트로 변환하는 함수를 정의합니다."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 자모 결합 함수\n",
    "def combine_jamo(jamo_text):\n",
    "    hangul_text = \"\"\n",
    "    buffer = []\n",
    "\n",
    "    # 자모 유효성 검사 함수\n",
    "    def is_valid_jamo(jamo):\n",
    "        return 0x1100 <= ord(jamo) <= 0x11FF\n",
    "\n",
    "    def flush_buffer(buffer):\n",
    "        if len(buffer) == 3:\n",
    "            initial = ord(buffer[0]) - 0x1100\n",
    "            medial = ord(buffer[1]) - 0x1161\n",
    "            final = ord(buffer[2]) - 0x11A7\n",
    "            if (0 <= initial < 19) and (0 <= medial < 21) and (0 <= final < 28):\n",
    "                return chr(0xAC00 + initial * 588 + medial * 28 + final)\n",
    "        elif len(buffer) == 2:\n",
    "            initial = ord(buffer[0]) - 0x1100\n",
    "            medial = ord(buffer[1]) - 0x1161\n",
    "            if (0 <= initial < 19) and (0 <= medial < 21):\n",
    "                return chr(0xAC00 + initial * 588 + medial * 28)\n",
    "        elif len(buffer) == 1:\n",
    "            return buffer[0]\n",
    "        return ''.join(buffer)\n",
    "\n",
    "    for jamo in jamo_text:\n",
    "        if is_jamo(jamo) and is_valid_jamo(jamo):\n",
    "            buffer.append(jamo)\n",
    "            if len(buffer) == 3:\n",
    "                hangul_text += flush_buffer(buffer)\n",
    "                buffer = []\n",
    "        else:\n",
    "            hangul_text += flush_buffer(buffer)\n",
    "            hangul_text += jamo\n",
    "            buffer = []\n",
    "\n",
    "    if buffer:\n",
    "        hangul_text += flush_buffer(buffer)\n",
    "\n",
    "    return hangul_text"
   ],
   "execution_count": 3,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 음성 파일로부터 텍스트 추론 함수\n",
    "음성 파일을 입력으로 받아서 모델을 통해 텍스트를 추론하는 함수를 정의합니다."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 음성 파일을 입력으로 받아 추론을 수행하는 함수\n",
    "def predict_from_audio(audio_path):\n",
    "    # 음성 파일 로드\n",
    "    speech_array, sampling_rate = torchaudio.load(audio_path)\n",
    "\n",
    "    # 샘플링 레이트가 16000Hz가 아닌 경우 리샘플링\n",
    "    if sampling_rate != 16000:\n",
    "        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)\n",
    "        speech_array = resampler(speech_array)\n",
    "\n",
    "    # 2D tensor로 변환\n",
    "    speech_array = speech_array.squeeze().numpy()\n",
    "\n",
    "    # 음성 파일 전처리\n",
    "    input_values = processor(speech_array, sampling_rate=16000, return_tensors=\"pt\").input_values\n",
    "    input_values = input_values.to(\"cuda\")\n",
    "\n",
    "    # 모델 추론\n",
    "    with torch.no_grad():\n",
    "        logits = model(input_values).logits\n",
    "\n",
    "    # 자모 텍스트 디코딩\n",
    "    pred_ids = torch.argmax(logits, dim=-1)\n",
    "    jamo_text = processor.batch_decode(pred_ids)[0]\n",
    "\n",
    "    # 자모 텍스트 결합하여 최종 문장 생성\n",
    "    final_text = combine_jamo(jamo_text)\n",
    "\n",
    "    return final_text"
   ],
   "execution_count": 4,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 예시: 음성 파일로부터 텍스트 추론\n",
    "지정된 음성 파일을 사용하여 텍스트를 추론합니다."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 예시: 음성 파일로부터 텍스트 추론\n",
    "audio_path = \"jiwon_.wav\"\n",
    "predicted_text = predict_from_audio(audio_path)\n",
    "print(\"Predicted Text: \", predicted_text)"
   ],
   "execution_count": 5,
   "outputs": [
    {
     "output_type": "stream",
     "text": [
      "Predicted Text:  예시 텍스트\n"
     ],
     "name": "stdout"
    }
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
