from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset


class SpeechDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal, sr = self.data[idx]
        input_dict = self.processor(signal, sampling_rate=sr, return_tensors="pt", padding=True, truncation=True)
        return input_dict


def train_model(train_data, snr_values, model_name='openai/whisper-large'):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    for snr in snr_values:
        noisy_audio = train_data[snr]
        dataset = SpeechDataset(noisy_audio, processor)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        model.train()
        for batch in dataloader:
            inputs = batch['input_ids'].squeeze(1)  # 取出输入
            labels = batch['texts'].squeeze(1)  # 取出标签
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 保存模型
        model.save_pretrained(f'whisper_model_snr_{snr}')
        processor.save_pretrained(f'whisper_model_snr_{snr}')


# 训练不同SNR下的模型
snr_values = list(train_data.keys())
train_model(train_data, snr_values)