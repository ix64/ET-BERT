## IX 的 复现日志

### 预训练

#### 构建词汇表
1. 获取作者使用预训练PCAP生成的语料文本
   - 语料结构：提取 PCAP Packet Payload，双字节 Hex编码，空格分隔；使得可以使用BPE分词器构建词汇表
   - [下载地址](https://drive.google.com/file/d/1P1Ru6my9QeJs0Mj6vGA4DyGJFXuI9_6t/view?usp=sharing)
2. 使用 `vocab_process/main.py` 脚本，构建词汇表
    - 注意修改 `word_dir` 与 `word_name` 的位置
    - 复现过程中，对其代码组织方式进行了少量修改，提高可读性，没有修改其功能
```bash
# 安装依赖
pip install scapy flowcontainer tokenizers
python vocab_process/main.py
```