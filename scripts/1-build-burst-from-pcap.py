#!/usr/bin/python3
# -*- coding:utf-8 -*-

import binascii
import logging
import os
import random
from os import path
from typing import List

import scapy.all as scapy


def preprocess(pcap_dir: str, word_filepath: str):
    logging.info("now pre-process pcap_dir is %s" % pcap_dir)

    packet_num = 0

    pcap_files: List[str] = []
    for parent, dirs, files in os.walk(pcap_dir):
        for file in files:
            file = str(file)
            if file.endswith(".pcapng"):
                continue
            pcap_files.append(path.join(parent, file))

    result_file = open(word_filepath, 'w')
    for n, file in enumerate(pcap_files):
        logging.info("processing pcap #%d: %s ..." % (n, file))
        packets = scapy.rdpcap(file)
        # word_packet = b''

        for p in packets:
            packet_num += 1

            word_packet = p.copy()  # ethernet frame
            words = (binascii.hexlify(bytes(word_packet)))  # raw ethernet bytes

            # 此处假设了捕获的 Packet: L2 Ethernet, L3 IPv4, L4 TCP
            # 38 字节即删除 Ethernet Header、IPv4 Header、以及 TCP 前4字节（即源端口和目标端口）
            # TODO[Code]: 处理方式不恰当，未考虑 VLAN、IPv6 等情况，应该修改为提取 TCP Packet 之后再进行裁剪
            words_string = words.decode()[76:]  # skip 38 bytes

            length = len(words_string)
            if length < 10:
                # 少于 5 bytes，则 TCP 包头不完整，跳过此包
                # TODO[Code]: 可以通过监测 TCP Payload 长度来检查
                continue

            for string_txt in cut(words_string, int(length / 2)):

                sentence = cut(string_txt, 1)
                for i in range(0, min(len(sentence), 256) - 1):
                    # 双字节 Hex 作为单词， 与上一句重叠
                    # TODO[Question]: 为社么要这么处理？
                    result_file.write(sentence[i])
                    result_file.write(sentence[i + 1])
                    result_file.write(' ')
                result_file.write("\n")
            result_file.write("\n")

    result_file.close()
    return packet_num


def cut(obj: str, sec: int):
    """
       按照字节进行切分成两半
       TODO[Code]: 优雅地实现
    """
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    remanent_count = len(result[0]) % 4
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result


def main():
    random.seed(40)
    pcap_dir = r"../download/cstnet-tls-1.3"
    word_filepath = r"../download/encrypted_traffic_burst.txt"
    preprocess(pcap_dir, word_filepath)


if __name__ == '__main__':
    main()
