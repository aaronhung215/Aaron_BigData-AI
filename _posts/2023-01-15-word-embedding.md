---
layout: post
title: Word Embedding
date: 2023-01-15
tags: wordembedding, vector
categories: wordembedding vector
comments: true
---

# Word Embedding
* Bag of words (BoW)
BoW 是一種將句子拆解成每個詞出現次數的方法，例如我將以下兩個句子拆解

```
I like apple.
{"I": 1, "like": 1, "apple": 1}
I like mango.
{"I": 1, "like": 1, "mango": 1}
```


這方法乍看之下分析出了句子的組成，但是卻忽略掉了很重要的次序關係，會使得以下兩句的結果完全相同

```
I like apple, but I don't like mango.
I like mango, but I don't like apple.
```

就bag of words而言，有如下缺點：1.沒有考慮到單詞的順序，2.忽略了單詞的語義資訊。因此這種方法對於短文字效果很差，對於長文字效果一般，通常在科研中用來做baseline



* Word Vector

> 使用一個向量來表示每一個詞(vector representation)，如此一來，就能把一段由許多詞組成的文句，轉換成一個個詞向量來表示，並把這樣數值化的資料，送到模型裡做後續的應用。
> 
> 一組好的詞向量，會使意思相似的詞在向量空間上比較靠近彼此，甚至詞義上的關聯可以用詞向量在空間中的關係來表示，如下圖所示

* Word embedding

> Word Embedding的概念。如果將word看作文本的最小單元，可以將Word Embedding理解為一種映射，其過程是：將文本空間中的某個word，通過一定的方法，映射或者說嵌入（embedding）到另一個數值向量空間（之所以稱之為embedding，是因為這種表示方法往往伴隨著一種降維的意思

- Frequency based embedding
基於頻率的Word Embedding又可細分為如下幾種：

    - Count Vector
    - TF-IDF Vector
    
    
- Predicted Based Embedding
基於預測的Word Embedding又可細分為如下幾種：

    - Word2vec
        - word2vec 提出了兩種訓練詞向量的方法，分別稱作 CBOW 和 Skipgram。
            - CBOW 模型輸入前後詞(contex word)詞向量，要預測出可能的目標詞(target word)， 將每個單詞的上下文作為輸入，並嘗試預測與上下文對應的單詞。考慮我們的例子： Have a great day.
        ![](https://i.imgur.com/sMeo9Dd.png)
            - Skipgram 模型則是輸入目標詞詞向量，預測出可能的前後詞。
            ![](https://i.imgur.com/hcxG6Qe.png)
        - 這裡說的預測是透過下圖所示的方法：輸入詞先透過查表(lookup)得到詞向量，接著通過一個矩陣(也可解釋為通過一層類神經網路)，預測共現(co-occur)的詞的機率。
        ![](https://i.imgur.com/Kxn9UoL.png)

    - Doc2Vec
    - GloVe
    
