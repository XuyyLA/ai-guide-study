"""Beam Search 算法演示

从零实现 Beam Search 算法，逐步展示搜索过程。
"""
import math


def beam_search(initial_sequence, beam_width, max_length, vocab, get_next_probs):
    beam = [(initial_sequence, 0.0)]  # (sequence, log_prob)
    completed = []

    for step in range(max_length):
        print(f"\n第 {step + 1} 步:")
        all_candidates = []
        for seq, score in beam:
            if seq.endswith('<eos>'):
                completed.append((seq, score))
                print(f"已完成序列: {seq}，得分为 {score}")
                continue
            next_probs = get_next_probs(seq)
            print(f"扩展序列: {seq}，当前得分为 {score}")
            for token, prob in next_probs.items():
                new_seq = seq + token
                new_score = score + math.log(prob)
                all_candidates.append((new_seq, new_score))
                print(f"  候选序列: {new_seq}，得分为 {new_score}")

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beam = all_candidates[:beam_width]

        print(f"\n选择的 {beam_width} 个顶束序列:")
        for seq, score in beam:
            print(f"  {seq}，得分为 {score}")

        if not beam:
            break

    completed += beam
    completed.sort(key=lambda x: x[1], reverse=True)

    print("\n已完成的所有序列:")
    for seq, score in completed:
        print(f"  {seq}，得分为 {score}")

    return completed[0][0]


def get_next_probs(seq):
    probs = {
        "": {"A": 0.4, "B": 0.3, "C": 0.2, "<eos>": 0.1},
        "A": {"A": 0.3, "B": 0.1, "C": 0.4, "<eos>": 0.2},
        "B": {"A": 0.1, "B": 0.1, "C": 0.3, "<eos>": 0.5},
        "AC": {"A": 0.1, "B": 0.2, "C": 0.5, "<eos>": 0.2},
    }
    return probs.get(seq, {"<eos>": 1.0})


initial_sequence = ""
beam_width = 2
max_length = 5
vocab = {"A", "B", "C", "<eos>"}

best_sequence = beam_search(initial_sequence, beam_width, max_length, vocab, get_next_probs)
print("\n最佳序列:", best_sequence)
