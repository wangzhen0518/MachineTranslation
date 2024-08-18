# TODO

1. lr scheduler
2. transformer 增加 memory mask
3. 增加 padding mask
    1. 分别传 casual mask 和 padding mask 是否会出现 nan
    2. 直接设 is_casual=True & attn_mask=None，是否会自动算 casual mask
4. need_weights -> False
5. 统一 mask 计算，subsequent mask 返回 2D 结果
6. padding mask, subsequent mask 分开
7. ~~切分训练集~~
8. 增加 validation 和 test
9. 增加 bleu 等 score 的计算
10. early stop
