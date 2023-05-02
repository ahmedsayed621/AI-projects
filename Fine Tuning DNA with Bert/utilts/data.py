# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd


class HF_dataset(torch.utils.data.Dataset):
    

    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.input_ids[index]),
            "attention_mask": torch.tensor(self.attention_masks[index]),
            "labels": torch.tensor(self.labels[index]),
        }


def val_dataset_generator(
    tokenizer,
    kmer_size,
    val_dir,
    maxl_len=512
):
    

    for file in os.listdir(val_dir):
        df_test = pd.read_csv(f"{val_dir}/{file}")
        print(file, len(df_test))
        val_kmers, labels_val = [], []

        cls = (
            "CLASS" if "CLASS" in df_test.columns else "Class"
        )  # in case the column name is "CLASS" or "Class" in the CSV file

        for seq, label in zip(df_test["SEQ"], df_test[cls]):
            kmer_seq = return_kmer(seq, K=kmer_size)
            val_kmers.append(kmer_seq)
            labels_val.append(label - 1)

        val_encodings = tokenizer.batch_encode_plus(
            val_kmers,
            max_length=maxl_len,  # max length of the sequences
            pad_to_max_length=True, # pad the sequences to the max length
            truncation=True, # truncate the sequences to the max length
            return_attention_mask=True, 
            return_tensors="pt", # return torch tensors
        )
        val_dataset = HF_dataset(
            val_encodings["input_ids"], val_encodings["attention_mask"], labels_val
        )
        yield val_dataset


def return_kmer(seq, K=3):
    """
    Example
    ----------
    >>> return_kmer("ATCGATCG", K=3)
    'ATC TCG CGA GAT ATC TCG'
    """

    kmer_list = []
    for x in range(len(seq) - K + 1):  # move a window of size K across the sequence
        kmer_list.append(seq[x : x + K])

    kmer_seq = " ".join(kmer_list)
    return kmer_seq


def is_dna_sequence(sequence):
    

    valid_bases = {"A", "C", "G", "T"}
    return all(base in valid_bases for base in sequence.upper())
