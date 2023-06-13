## IX 的 复现日志

### 预训练

#### 构建语料文本

1. 从 PCAP 生成语料文本
    - 复现过程中，对其代码组织方式进行了少量修改，提高可读性，没有修改其功能
    - 语料结构：提取 PCAP Packet Payload，双字节 Hex编码，重复一次，空格分隔（使得可以使用BPE分词器构建词汇表）
    - 疑问：为什么要重叠地重复一次
    ```bash
    pip isntall scapy
    python scripts/1-build-burst-from-pcap.py
    ```

2. 获取作者使用预训练PCAP生成的语料文本
    - [下载地址](https://drive.google.com/file/d/1P1Ru6my9QeJs0Mj6vGA4DyGJFXuI9_6t/view?usp=sharing)

#### 构建词汇表

1. 使用 `vocab_process/main.py` 脚本，构建词汇表
    - 注意修改 `word_dir` 与 `word_name` 的位置
    - 复现过程中，对其代码组织方式进行了少量修改，提高可读性，没有修改其功能

```bash
# 安装依赖
pip install tokenizers
python scripts/2-build-vocab-from-burst.py
```

#### 构建数据集

1. 将 **语料文本** 转换为使用 **词汇表** 表示的 **数据集**

```bash
pip install -r requirements.txt
pip install numpy
python preprocess.py \
  --corpus_path corpora/encrypted_traffic_burst.txt \
  --vocab_path models/encryptd_vocab.txt \
  --dataset_path dataset.pt \
  --processes_num $(nproc) \
  --target bert
```

#### 进行预训练

- 模型设定
    - Embedding: 词嵌入，word_pos_seg (即 BERT embedding)，包含 Word & Position & Segment 的 Embedding
    - Encoder: 编码器，使用 BERT Transformer
    - Target: Finetune 阶段可替换的，预训练阶段 使用 BERT Target，MLM & NSP
- 无监督学习所有流量
    - nsp: Next sentence prediction, 根据上文预测下一个 Token
    - mlm: Masked language model, 根据上下文预测 Masked Token
- 显存不能跑太满，运行一段时间后可能会增加

```bash
# 2 x TITAN Xp 
export PYTHONPATH=.
python pre-training/pretrain.py \
  --dataset_path dataset.pt \
  --vocab_path models/encryptd_vocab.txt \
  --pretrained_model_path models/train-1/pre-trained_model.bin-110000 \
  --output_model_path     models/train-2/pre-trained_model.bin \
  --world_size 2 \
  --gpu_ranks 0 1 \
  --total_steps 500000 \
  --save_checkpoint_steps 10000 \
  --batch_size 48 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --target bert
```

```bash
# 1 x 2060S
python pre-training/pretrain.py \
  --dataset_path dataset.pt \
  --vocab_path models/encryptd_vocab.txt \
  --pretrained_model_path models/train-1/pre-trained_model.bin \
  --output_model_path models/pre-trained_model.bin \
  --world_size 1 \
  --gpu_ranks 0 \
  --total_steps 500000 \
  --save_checkpoint_steps 10000 \
  --batch_size 16 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --target bert
```

### Finetune (CSTNET TLS 1.3 Packet)

#### 构建数据集

- 对脚本 `data_process/dataset_generation.py` 进行了一些改动
    - 修复可读性、安全性、兼容性问题

```bash

```

#### 执行 Finetune

- 模型：使用 Classifier 替换 target 进行 finetune

```bash
python3 fine-tuning/run_classifier.py \
  --pretrained_model_path models/train-1/pre-trained_model.bin \
  --vocab_path            models/encryptd_vocab.txt \
  --train_path            datasets/cstnet-tls1.3/packet/train_dataset.tsv \
  --dev_path              datasets/cstnet-tls1.3/packet/valid_dataset.tsv \
  --test_path             datasets/cstnet-tls1.3/packet/test_dataset.tsv \
  --epochs_num 10 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5
```

#### 运行推理

- 推理脚本需要引用 fine-tuning 定义的 Classifier，因此暂时移动到 fine-tuning 之内

```bash
python3 fine-tuning/run_classifier_infer.py \
  --load_model_path models/finrtune-1/pre-trained_model.bin \
  --vocab_path      models/encryptd_vocab.txt \
  --test_path       datasets/cstnet-tls1.3/packet/nolabel_test_dataset.tsv \
  --prediction_path datasets/cstnet-tls1.3/packet/prediction.tsv \
  --labels_num      120 \
  --embedding       word_pos_seg \
  --encoder         transformer \
  --mask            fully_visible
```

