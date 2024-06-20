{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Wav2Vec2 Fine-Tuning for Korean Speech Recognition\n",
                "This notebook demonstrates the fine-tuning of the Wav2Vec2 model for Korean speech recognition using the Zeroth-Korean dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install transformers[torch] accelerate -U\n",
                "!pip install datasets torchaudio -U\n",
                "!pip install jiwer jamo\n",
                "!pip install tensorboard"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Preprocessing"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "from datasets import load_dataset\n",
                "import re\n",
                "from jamo import h2j, j2hcj\n",
                "\n",
                "# Load Zeroth-Korean dataset\n",
                "dataset = load_dataset('zeroth_korean', 'clean')\n",
                "\n",
                "# Text cleaning function\n",
                "def clean_text(text):\n",
                "    text = re.sub(r'[^ \u3131-\u3163\uac00-\ud7a3]', '', text)\n",
                "    text = re.sub(r'\\s+', ' ', text)\n",
                "    return text.strip()\n",
                "\n",
                "# Apply text cleaning and Jamo separation\n",
                "def prepare_data(batch):\n",
                "    batch['text'] = clean_text(batch['text'])\n",
                "    batch['text'] = j2hcj(h2j(batch['text']))\n",
                "    return batch\n",
                "\n",
                "dataset = dataset.map(prepare_data)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Tokenizer and Vocabulary"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor\n",
                "\n",
                "# Define the tokenizer\n",
                "vocab_dict = {c: i for i, c in enumerate(set(''.join(dataset['train']['text'])))}\n",
                "vocab_dict['[PAD]'] = len(vocab_dict)\n",
                "tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(vocab_dict)\n",
                "\n",
                "# Define the processor\n",
                "processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Model Configuration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import Wav2Vec2ForCTC\n",
                "\n",
                "# Load the Wav2Vec2 model\n",
                "model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base')\n",
                "\n",
                "# Configure the model\n",
                "model.config.update({\n",
                "    'attention_dropout': 0.0,\n",
                "    'hidden_dropout': 0.0,\n",
                "    'feat_proj_dropout': 0.0,\n",
                "    'mask_time_prob': 0.05,\n",
                "    'gradient_checkpointing': True\n",
                "})"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Training"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import Trainer, TrainingArguments\n",
                "\n",
                "# Training arguments\n",
                "training_args = TrainingArguments(\n",
                "    output_dir='./results',\n",
                "    group_by_length=True,\n",
                "    per_device_train_batch_size=32,\n",
                "    evaluation_strategy='steps',\n",
                "    num_train_epochs=10,\n",
                "    gradient_accumulation_steps=2,\n",
                "    fp16=True,\n",
                "    save_steps=500,\n",
                "    eval_steps=500,\n",
                "    logging_steps=500,\n",
                "    learning_rate=1e-4,\n",
                "    warmup_steps=500,\n",
                "    save_total_limit=2,\n",
                ")\n",
                "\n",
                "# Trainer\n",
                "trainer = Trainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    data_collator=data_collator,\n",
                "    train_dataset=dataset['train'],\n",
                "    eval_dataset=dataset['validation'],\n",
                "    tokenizer=processor.feature_extractor,\n",
                ")\n",
                "\n",
                "# Train the model\n",
                "trainer.train()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "from datasets import load_metric\n",
                "\n",
                "# Load CER metric\n",
                "cer = load_metric('cer')\n",
                "\n",
                "# Evaluate the model\n",
                "results = trainer.evaluate()\n",
                "print(f'CER: {results['eval_cer']}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Inference"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor\n",
                "import torchaudio\n",
                "\n",
                "# Load the model and processor\n",
                "model_name = 'Kkonjeong/wav2vec2-base-korean'\n",
                "model = Wav2Vec2ForCTC.from_pretrained(model_name)\n",
                "processor = Wav2Vec2Processor.from_pretrained(model_name)\n",
                "\n",
                "# Perform inference on an audio file\n",
                "def predict(file_path):\n",
                "    # Load and preprocess the audio file\n",
                "    speech_array, sampling_rate = torchaudio.load(file_path)\n",
                "    if sampling_rate != 16000:\n",
                "        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)\n",
                "        speech_array = resampler(speech_array)\n",
                "    input_values = processor(speech_array.squeeze().numpy(), sampling_rate=16000).input_values[0]\n",
                "    input_values = torch.tensor(input_values).unsqueeze(0).to('cuda')\n",
                "    \n",
                "    # Get model predictions\n",
                "    with torch.no_grad():\n",
                "        logits = model(input_values).logits\n",
                "    predicted_ids = torch.argmax(logits, dim=-1)\n",
                "    transcription = processor.batch_decode(predicted_ids)[0]\n",
                "    return transcription\n",
                "\n",
                "audio_file_path = 'jiwon_.wav'\n",
                "transcription = predict(audio_file_path)\n",
                "print('Transcription:', transcription)"
            ]
        }
    ],
    "metadata": {
        "colab": {
            "collapsed_sections": [],
            "name": "wav2vec2_finetuning.ipynb"
        },
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
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}
