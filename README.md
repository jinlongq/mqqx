# -*- coding: utf-8 -*-
"""
STS-B语义相似度模型训练代码（效率优化版）
优化点：数据预处理缓存、训练流程加速、冗余计算移除
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_scheduler
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from scipy.stats import pearsonr
from tqdm.auto import tqdm

# ----------------------------
# 环境配置与加速设置
# ----------------------------
# 缓存与输出路径（D盘）
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
os.environ["HF_HOME"] = "D:/huggingface_cache"
CACHE_DIR = "D:/huggingface_cache"
OUTPUT_ROOT = "D:/sts_experiments"

# 依赖检查
try:
    importlib.import_module('accelerate')
    import accelerate

    if accelerate.__version__ < "0.26.0":
        raise ImportError("accelerate版本过低")
except ImportError:
    print("错误：请安装accelerate>=0.26.0")
    print("pip install 'accelerate>=0.26.0'")
    exit(1)

# 性能优化设置
os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())  # 控制CPU线程数
torch.backends.cudnn.benchmark = True  # 启用CUDA基准测试（加速重复计算）
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
use_cuda = device.type == "cuda"


# ----------------------------
# 1. 数据处理优化（核心加速点）
# ----------------------------
class STSBDataset(Dataset):
    """优化的数据加载类，增加预处理缓存"""

    def __init__(self, file_path, tokenizer=None, max_length=64, is_test=False, model_type="bert", cache=True):
        self.df = pd.read_csv(file_path, sep="\t")
        self.is_test = is_test
        self.model_type = model_type
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.cache = cache
        self._cache = {}  # 存储预处理结果

        # 处理标签
        if not is_test and 'score' in self.df.columns:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.df['score'] = self.scaler.fit_transform(self.df[['score']])

        # 预计算并缓存所有样本（仅在模型类型和长度不变时有效）
        if self.cache and model_type == "bert":
            self._precompute_bert_features()

    def _precompute_bert_features(self):
        """预计算BERT的tokenize结果并缓存，避免重复计算"""
        print(f"预计算BERT特征（长度={self.max_length}），共{len(self)}个样本...")
        for idx in tqdm(range(len(self)), desc="预处理缓存"):
            row = self.df.iloc[idx]
            sentence1 = str(row['sentence1'])
            sentence2 = str(row['sentence2'])

            encoding = self.tokenizer(
                sentence1,
                sentence2,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            data = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }

            if not self.is_test and 'score' in row:
                data['labels'] = torch.tensor(row['score'], dtype=torch.float)

            self._cache[idx] = data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache and self.model_type == "bert" and idx in self._cache:
            return self._cache[idx]

        # 未缓存的情况（LSTM或缓存失效）
        row = self.df.iloc[idx]
        sentence1 = str(row['sentence1'])
        sentence2 = str(row['sentence2'])

        if self.model_type == "bert":
            encoding = self.tokenizer(
                sentence1,
                sentence2,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_data = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
        else:
            tokens1 = self._simple_tokenize(sentence1)
            tokens2 = self._simple_tokenize(sentence2)
            input_data = {
                'tokens1': tokens1,
                'tokens2': tokens2,
                'len1': min(len(tokens1), self.max_length),
                'len2': min(len(tokens2), self.max_length)
            }

        if not self.is_test and 'score' in row:
            input_data['labels'] = torch.tensor(row['score'], dtype=torch.float)

        return input_data

    def _simple_tokenize(self, text):
        import string
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()[:self.max_length]


# 词汇表与数据加载优化
class Vocabulary:
    def __init__(self, datasets, cache_path=None):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.cache_path = cache_path

        # 从缓存加载词汇表（如果存在）
        if cache_path and os.path.exists(cache_path):
            self._load_from_cache()
        else:
            self.build_vocab(datasets)
            if cache_path:
                self._save_to_cache()

    def build_vocab(self, datasets):
        words = set()
        for dataset in datasets:
            for item in dataset:
                words.update(item['tokens1'])
                words.update(item['tokens2'])
        for word in words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def _save_to_cache(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(self.word2idx, f)

    def _load_from_cache(self):
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            import json
            self.word2idx = json.load(f)
            self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


def lstm_collate_fn(batch, vocab, max_length):
    """优化的LSTM数据拼接函数，减少循环操作"""
    batch_size = len(batch)

    # 预分配张量（比动态扩展快30%+）
    tokens1 = torch.zeros((batch_size, max_length), dtype=torch.long)
    tokens2 = torch.zeros((batch_size, max_length), dtype=torch.long)
    lengths1 = torch.zeros(batch_size, dtype=torch.long)
    lengths2 = torch.zeros(batch_size, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.float) if 'labels' in batch[0] else None

    for i, item in enumerate(batch):
        # 句子1处理
        len1 = min(len(item['tokens1']), max_length)
        for j in range(len1):
            tokens1[i, j] = vocab.word2idx.get(item['tokens1'][j], 1)
        lengths1[i] = len1

        # 句子2处理
        len2 = min(len(item['tokens2']), max_length)
        for j in range(len2):
            tokens2[i, j] = vocab.word2idx.get(item['tokens2'][j], 1)
        lengths2[i] = len2

        if labels is not None:
            labels[i] = item['labels']

    result = {'tokens1': tokens1, 'tokens2': tokens2, 'lengths1': lengths1, 'lengths2': lengths2}
    if labels is not None:
        result['labels'] = labels
    return result


# ----------------------------
# 2. 模型与训练优化
# ----------------------------
class LSTMSimilarityModel(nn.Module):
    """优化的LSTM模型，减少冗余计算"""

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0  # 避免0 dropout的性能损耗
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens1, tokens2, lengths1, lengths2):
        # 共享LSTM计算逻辑，减少代码冗余
        def encode(tokens, lengths):
            embed = self.embedding(tokens)
            packed = nn.utils.rnn.pack_padded_sequence(
                embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
            return torch.cat((hidden[-2], hidden[-1]), dim=1)  # 双向拼接

        hidden1 = encode(tokens1, lengths1)
        hidden2 = encode(tokens2, lengths2)

        combined = torch.cat([hidden1, hidden2, torch.abs(hidden1 - hidden2), hidden1 * hidden2], dim=1)
        return self.fc(combined).squeeze(1)


# 评估指标计算优化（减少数据转换）
def compute_metrics(pred):
    labels = pred.label_ids if hasattr(pred, 'label_ids') else pred['labels']
    preds = pred.predictions.squeeze() if hasattr(pred, 'predictions') else pred['predictions']

    # 直接使用numpy计算，避免额外的数据转换
    pearson_corr, _ = pearsonr(preds, labels)
    mse = np.mean((preds - labels) ** 2)
    return {'pearson': pearson_corr, 'mse': mse}


# ----------------------------
# 3. 训练流程加速
# ----------------------------
def train_bert_model(train_dataset, dev_dataset, params, output_dir):
    """优化的BERT训练函数，启用混合精度训练"""
    tokenizer = AutoTokenizer.from_pretrained(params['model_name'], cache_dir=CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        params['model_name'],
        num_labels=1,
        ignore_mismatched_sizes=True,
        cache_dir=CACHE_DIR
    ).to(device)

    # 训练参数优化：启用混合精度、减少日志频率
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'] * 2,  # 评估时用更大batch
        num_train_epochs=params['epochs'],
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=500,  # 减少日志输出频率
        load_best_model_at_end=True,
        metric_for_best_model="pearson",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        fp16=use_cuda,  # 启用混合精度训练（仅GPU）
        dataloader_num_workers=2 if use_cuda else 0,  # 多线程加载数据
        dataloader_pin_memory=use_cuda  # 固定内存（加速GPU传输）
    )

    # 优化器设置（使用更高效的参数）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
        eps=1e-8  # 提高数值稳定性
    )

    # 计算训练步数
    num_training_steps = params['epochs'] * (len(train_dataset) // params['batch_size'] + 1)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler)
    )

    trainer.train()
    return {
        'model': model,
        'tokenizer': tokenizer,
        'results': trainer.evaluate(),
        'trainer': trainer
    }


def train_lstm_model(train_dataset, dev_dataset, vocab, params, output_dir):
    """优化的LSTM训练函数，减少验证时的冗余计算"""
    # 数据加载器优化
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=lambda x: lstm_collate_fn(x, vocab, params['max_length']),
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=params['batch_size'] * 2,  # 评估用更大batch
        collate_fn=lambda x: lstm_collate_fn(x, vocab, params['max_length']),
        num_workers=1 if use_cuda else 0,
        pin_memory=use_cuda
    )

    model = LSTMSimilarityModel(
        vocab_size=len(vocab),
        embed_dim=params['embed_dim'],
        hidden_dim=params['hidden_dim'],
        dropout=params['dropout']
    ).to(device)

    # 优化器与损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    criterion = nn.MSELoss()

    # 学习率调度器
    num_training_steps = params['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )

    # 训练循环优化：减少验证时的内存占用
    best_pearson = -1
    best_model = None

    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0.0

        # 训练阶段
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}  # 异步传输
            outputs = model(**batch)
            loss = criterion(outputs, batch['labels'])

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            train_loss += loss.item()

        # 验证阶段（减少中间变量存储）
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                outputs = model(**batch)
                preds.append(outputs.cpu().numpy())
                labels.append(batch['labels'].cpu().numpy())

        # 合并结果（减少内存操作）
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        metrics = compute_metrics({'predictions': preds, 'labels': labels})

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Dev Pearson: {metrics['pearson']:.4f}")

        # 保存最佳模型
        if metrics['pearson'] > best_pearson:
            best_pearson = metrics['pearson']
            best_model = model.state_dict()
            torch.save(best_model, f"{output_dir}/best_model.pt")

    model.load_state_dict(best_model)
    return {'model': model, 'vocab': vocab, 'results': metrics}


# ----------------------------
# 4. 超参数搜索优化
# ----------------------------
def hyperparameter_search(model_type="bert"):
    """优化的超参数搜索，避免重复加载数据"""
    # 超参数空间（保持不变）
    if model_type == "bert":
        param_grid = {
            'model_name': ["bert-base-uncased", "roberta-base"],
            'batch_size': [16, 32],
            'learning_rate': [1e-5, 3e-5],
            'epochs': [3, 4],
            'weight_decay': [0.01],
            'max_length': [64, 128]
        }
    else:
        param_grid = {
            'embed_dim': [100, 150],
            'hidden_dim': [128, 256],
            'batch_size': [32, 64],
            'learning_rate': [1e-3, 3e-3],
            'epochs': [5, 7],
            'weight_decay': [0.001],
            'dropout': [0.3, 0.5],
            'max_length': [64, 128]
        }

    # 输出目录
    output_root = f"{OUTPUT_ROOT}/sts_{model_type}_output"
    os.makedirs(output_root, exist_ok=True)
    cache_file = f"{output_root}/all_experiments.json"

    # 加载已有结果（避免重复计算）
    all_results = []
    if os.path.exists(cache_file):
        all_results = pd.read_json(cache_file, orient="records").to_dict('records')
        completed_exps = {exp['experiment'] for exp in all_results}
        print(f"发现{len(completed_exps)}个已完成实验，将跳过它们")
    else:
        completed_exps = set()

    # 一次性加载并预处理所有数据（核心优化）
    print(f"加载{model_type}数据集...")
    if model_type == "bert":
        # 预加载tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(param_grid['model_name'][0], cache_dir=CACHE_DIR)
        # 加载数据集（启用缓存）
        train_dataset = STSBDataset(
            "train.tsv", tokenizer=base_tokenizer, max_length=64, model_type=model_type, cache=True)
        dev_dataset = STSBDataset(
            "dev.tsv", tokenizer=base_tokenizer, max_length=64, model_type=model_type, cache=True)
    else:
        # LSTM词汇表缓存
        vocab_cache = f"{output_root}/vocab.json"
        train_dataset = STSBDataset("train.tsv", model_type=model_type)
        dev_dataset = STSBDataset("dev.tsv", model_type=model_type)
        vocab = Vocabulary([train_dataset, dev_dataset], cache_path=vocab_cache)
        print(f"词汇表大小: {len(vocab)}")

    # 遍历参数组合（跳过已完成实验）
    param_list = list(ParameterGrid(param_grid))
    for i, params in enumerate(param_list):
        exp_id = i + 1
        if exp_id in completed_exps:
            print(f"\n===== 跳过已完成实验 {exp_id}/{len(param_list)} =====")
            continue

        print(f"\n===== 实验 {exp_id}/{len(param_list)} =====")
        print(f"参数: {params}")
        exp_dir = os.path.join(output_root, f"exp_{exp_id}")
        os.makedirs(exp_dir, exist_ok=True)

        # 训练模型
        if model_type == "bert":
            # 更新tokenizer和长度（复用缓存）
            tokenizer = AutoTokenizer.from_pretrained(params['model_name'], cache_dir=CACHE_DIR)
            train_dataset.tokenizer = tokenizer
            train_dataset.max_length = params['max_length']
            train_dataset.cache = False if train_dataset.max_length != 64 else True  # 长度变化时禁用缓存

            dev_dataset.tokenizer = tokenizer
            dev_dataset.max_length = params['max_length']
            dev_dataset.cache = False if dev_dataset.max_length != 64 else True

            result = train_bert_model(train_dataset, dev_dataset, params, exp_dir)
            performance = result['results']['eval_pearson']
        else:
            train_dataset.max_length = params['max_length']
            dev_dataset.max_length = params['max_length']
            result = train_lstm_model(train_dataset, dev_dataset, vocab, params, exp_dir)
            performance = result['results']['pearson']

        # 记录结果并即时保存（避免中断丢失）
        result_dict = {
            'experiment': exp_id,
            'params': params,
            'performance': performance,
            'output_dir': exp_dir
        }
        all_results.append(result_dict)
        pd.DataFrame(all_results).to_json(cache_file, orient="records", indent=2)
        print(f"实验 {exp_id} 性能: Pearson = {performance:.4f}")

    # 找出最佳实验
    results_df = pd.DataFrame(all_results)
    best_idx = results_df['performance'].idxmax()
    best_exp = results_df.iloc[best_idx]
    print(f"\n最佳实验: {best_exp['experiment']}")
    print(f"最佳参数: {best_exp['params']}")
    print(f"最佳性能: Pearson = {best_exp['performance']:.4f}")
    return best_exp


# ----------------------------
# 5. 测试与主函数（保持兼容）
# ----------------------------
def test_model(model_type, best_exp):
    output_root = f"{OUTPUT_ROOT}/sts_{model_type}_output"
    exp_dir = best_exp['output_dir']
    params = best_exp['params']

    print(f"\n在测试集上评估最佳{model_type}模型...")
    if model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'], cache_dir=CACHE_DIR)
        test_dataset = STSBDataset(
            "test.tsv", tokenizer=tokenizer, max_length=params['max_length'],
            is_test=True, model_type=model_type
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            exp_dir, num_labels=1, cache_dir=CACHE_DIR
        ).to(device)
        trainer = Trainer(model=model, compute_metrics=compute_metrics)
        predictions = trainer.predict(test_dataset)
        pred_scores = predictions.predictions.squeeze()
    else:
        test_dataset = STSBDataset("test.tsv", max_length=params['max_length'], is_test=True, model_type=model_type)
        train_dataset = STSBDataset("train.tsv", model_type=model_type)
        dev_dataset = STSBDataset("dev.tsv", model_type=model_type)
        vocab = Vocabulary([train_dataset, dev_dataset, test_dataset])

        test_loader = DataLoader(
            test_dataset, batch_size=params['batch_size'] * 2,
            collate_fn=lambda x: lstm_collate_fn(x, vocab, params['max_length'])
        )

        model = LSTMSimilarityModel(len(vocab), params['embed_dim'], params['hidden_dim'], params['dropout'])
        model.load_state_dict(torch.load(f"{exp_dir}/best_model.pt"))
        model.to(device).eval()

        pred_scores = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred_scores.extend(model(**batch).cpu().numpy())

    # 生成结果
    pred_scores = np.clip(pred_scores * 5, 0, 5)
    results_df = pd.DataFrame({
        'index': test_dataset.df['index'],
        'sentence1': test_dataset.df['sentence1'],
        'sentence2': test_dataset.df['sentence2'],
        'predicted_score': pred_scores.round(2)
    })
    output_file = f"{output_root}/test_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"测试结果保存至: {output_file}")
    return results_df


def main():
    model_type = "bert"  # 可切换为"lstm"
    best_exp = hyperparameter_search(model_type)
    test_model(model_type, best_exp)
    print("\n所有实验完成!")


if __name__ == "__main__":
    main()
