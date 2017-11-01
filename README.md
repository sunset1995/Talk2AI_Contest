# 科技部科技大擂台 與AI對話 熱身賽

隊名：現充與現充與魯蛇
隊員：[ChiWeiHsiao](https://github.com/ChiWeiHsiao), [sunset](https://github.com/sunset1995), [SouthRa](https://github.com/ko19951231)

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

最後版本：選 variance 高的多個 model 的答案做 voting
- dual-lstm (dual-lstm 的 rnn-encoder 差異性較小，但 dual-lstm 表現較好故選之)
- 多個 train deep model 時 fine-tuned 過的 word2vec 跑 naive 方法，總票數比 dual-lstm 少
- smn 沒時間 tune 跟 train


熱身賽決賽預賽結果：  

| 隊伍 | 語音分數(現場運氣) | 文本分數 | 總分 |
| --- | :--- | :--- | :--- |
| 現充與現充與魯蛇 | 0.6756 | 0.5810 | 0.6094 |
| other team | 0.7567 | 0.5185 | 0.5899 |
| other team | 0.5945 | 0.4955 | 0.5252 |
| other team | 0.5945 | 0.4890 | 0.5206 |
| other team | 0.6216 | 0.4725 | 0.5172 |
| other team | 0.5945 | 0.4695 | 0.5070 |
| other team | 0.4864 | 0.4780 | 0.4805 |
| other team | 0.4864 | 0.4565 | 0.4654 |
| other team | 0.4864 | 0.4500 | 0.4609 |
| other team | 0.5135 | 0.4320 | 0.4564 |


熱身賽決賽四強結果：  

| 隊伍 | 語音分數(現場運氣) | 文本分數 | 總分 |
| --- | :--- | :--- | :--- |
| 現充與現充與魯蛇 | 0.6101 | 0.5875 | 0.5943 |
| other team | 0.4915 | 0.5190 | 0.5107 |
| other team | 0.4915 | 0.5105 | 0.5048 |
| other team | 0.4237 | 0.5135 | 0.4865 |

