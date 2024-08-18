# TODO

1. lr scheduler
2. transformer 增加 memory mask
3. 增加 padding mask
    1. ~~分别传 casual mask 和 padding mask 是否会出现 nan~~, 会
    2. ~~直接设 is_casual=True & attn_mask=None，是否会自动算 casual mask~~， 不会
4. ~~need_weights -> False，https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html~~, pytorch 实现的 transformer 中就是 False
5. ~~统一 mask 计算，subsequent mask 返回 2D 结果~~
6. ~~padding mask, subsequent mask 分开~~
7. ~~切分训练集~~
8. 增加 validation 和 test
9. 增加 bleu 等 score 的计算
10. early stop
11. 修改传入 device 的位置，特别是 batch 中
12. nan 问题
    1.  ~~如何让 tokenizer 在句尾增加 padding~~, `tokenizer.padding_side="right"`
    2.  https://github.com/pytorch/pytorch/issues/41508, 不填充 float('-inf'), 改为填充一个很小的负数，如 -1e9
    3.  ~~传入 float mask 而不是 bool mask，这样 pytorch 不对 mask 进行修改，使用的是 float mask 原始值，目前使用的是 `-1e8`，就不会出现 nan 了~~
13. 断电续接训练
14. 修正 Transformer 实现，增加各个 mask 部分
15. 由于 mask 已修改，所以需要修正 translate 部分。
