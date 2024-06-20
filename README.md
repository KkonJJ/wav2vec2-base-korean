
# Fine-Tuning Wav2Vec2-Base for Korean Speech Recognition

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Card-orange)](https://huggingface.co/Kkonjeong/wav2vec2-base-korean/blob/main/README.md)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rVhqR22zrlSWXGbrpux3i-dQlR8uqSc4#scrollTo=Jj6TydLiDpCw)

This project focuses on fine-tuning Facebook's Wav2Vec2-Base model for Korean speech recognition using the Zeroth-Korean dataset.

## Project Purpose

The main goal of this project is to develop a robust Korean speech recognition model by leveraging the power of Wav2Vec2, a state-of-the-art self-supervised learning model for automatic speech recognition (ASR). By fine-tuning this model with the Zeroth-Korean dataset, we aim to improve its performance in understanding and transcribing Korean speech accurately.

## Setup and Requirements

To replicate this project, you'll need to install the following packages:

```bash
!pip install transformers[torch] accelerate -U
!pip install datasets torchaudio -U
!pip install jiwer jamo
!pip install tensorboard
```

## Methodology

### Data Preprocessing

1. **Dataset Loading**: The Zeroth-Korean dataset is loaded using the datasets library.
2. **Text Cleaning**: Special characters are removed from the dataset to standardize the text.
3. **Jamo Separation**: Korean characters are separated into Jamo (Korean alphabet components) to facilitate the training process.

### Tokenizer and Vocabulary

A custom tokenizer is created using a vocabulary that includes all possible Jamo characters along with special tokens.

### Model Configuration

The Wav2Vec2-Base model is configured with specific parameters for Korean speech recognition. Key configurations include:

- Attention dropout, hidden dropout, and feature projection dropout set to 0.0.
- Mask time probability set to 0.05.
- Gradient checkpointing enabled for efficient memory usage during training.

### Training

The model is trained using the Trainer API from the transformers library. Key training configurations include:

- Batch size of 32.
- 10 epochs.
- Learning rate of 1e-4.
- Evaluation at every 500 steps.

### Evaluation

The model's performance is evaluated using the Character Error Rate (CER) metric.

## Results

The fine-tuned model achieved a test CER of 0.073, demonstrating its capability in accurately transcribing Korean speech.

| Step | Training Loss | Validation Loss | CER |
|------|----------------|-----------------|-----|
| 500  | 3.601800       | 1.046800        | 0.268646 |
| 1000 | 0.594000       | 0.494357        | 0.156528 |
| 1500 | 0.393300       | 0.406724        | 0.132043 |
| 2000 | 0.313800       | 0.338634        | 0.116344 |
| 2500 | 0.256700       | 0.307439        | 0.105724 |
| 3000 | 0.223100       | 0.279376        | 0.097198 |
| 3500 | 0.193500       | 0.271789        | 0.091062 |
| 4000 | 0.165500       | 0.248423        | 0.084631 |
| 4500 | 0.147400       | 0.235357        | 0.082036 |
| 5000 | 0.131800       | 0.236439        | 0.079886 |
| 5500 | 0.119000       | 0.233483        | 0.076642 |
| 6000 | 0.107500       | 0.229132        | 0.075085 |
| 6500 | 0.099200       | 0.226362        | 0.073195 |

## Inference

The fine-tuned model can be used to transcribe Korean speech by loading the model and processor and passing audio files through them.

### Example Usage

```python
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio

# Load the model and processor
model_name = "Kkonjeong/wav2vec2-base-korean"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Perform inference on an audio file
def predict(file_path):
    # Load and preprocess the audio file
    speech_array, sampling_rate = torchaudio.load(file_path)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)
    input_values = processor(speech_array.squeeze().numpy(), sampling_rate=16000).input_values[0]
    input_values = torch.tensor(input_values).unsqueeze(0).to("cuda")
    
    # Get model predictions
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

audio_file_path = "jiwon_.wav"
transcription = predict(audio_file_path)
print("Transcription:", transcription)
```

## Conclusion

This project successfully fine-tuned the Wav2Vec2-Base model for Korean speech recognition. The model demonstrated a low Character Error Rate (CER) on the test dataset, indicating its effectiveness in transcribing Korean speech.

The fine-tuned model and processor have been uploaded to Hugging Face Hub for public use.

## References

- Hugging Face Wav2Vec2
- Zeroth-Korean Dataset
