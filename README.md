# 科技部科技大擂台 與AI對話

隊員：[ChiWeiHsiao](https://github.com/ChiWeiHsiao), [SouthRa](https://github.com/ko19951231), [sunset](https://github.com/sunset1995)

我們嘗試過的方法:
- `naive`
    - count 下句的詞在上句出現幾次
    - 將一句話 word2vec 後的 vector 取(加權)平均，算上下句的 cosine similarity
- `dual-lstm`
    - 兩個不同的 rnn encode 上下句後，`sigmoid(u, Mv)` 來 train 其認識上下句
    - 用 `sigmoid(u, Mv)` 來代表為上下句的機率
    - 用 `u`, `v` 的 cosine similarity
- `rnn-encoder`
    - 同 `dual-lstm` 但上下句的 rnn 為 shared weights
- `smn`
    - [Sequential Matching Network: A New Architecture for Multi-turn
Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1612.01627.pdf)
