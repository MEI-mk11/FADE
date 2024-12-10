from sklearn.metrics import accuracy_score


def test_model(model, processor, test_data, snr_values):
    rrm_matrix = np.zeros((len(snr_values), len(snr_values)))

    for train_snr_idx, train_snr in enumerate(snr_values):
        for test_snr_idx, test_snr in enumerate(snr_values):
            model = WhisperForConditionalGeneration.from_pretrained(f'whisper_model_snr_{train_snr}')
            processor = WhisperProcessor.from_pretrained(f'whisper_model_snr_{train_snr}')

            sentences = test_data[test_snr]
            pred_labels = []
            true_labels = []
            for signal, sr in sentences:
                inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True, truncation=True)
                generated_ids = model.generate(inputs['input_ids'])
                decoded_output = processor.decode(generated_ids[0], skip_special_tokens=True)

                # true_labels 应为原始的音频文本（可以从实际音频的标签中获得）
                true_labels.append("真实文本")  # 需要替换成实际的标签
                pred_labels.append(decoded_output)

            # 计算识别率
            accuracy = accuracy_score(true_labels, pred_labels)
            rrm_matrix[train_snr_idx, test_snr_idx] = accuracy

    return rrm_matrix


# 测试不同信噪比下的模型
rrm_matrix = test_model(model, processor, train_data, snr_values)