bert few shot learning

use bert pre-trained model, fine tune few shot learning. 

主要目的用来 【语义匹配】

[bert模型可以做相似度任务吗？](https://www.zhihu.com/question/354129879)

发现什么-1层，-2层的bert句向量都不好使，压根没法用，都很高，没区分性，最最重要的的是相似度排序还不行。

毕竟任务不一样，不能简单相加、平均、取[cls]啥的。

还是需要一些下游任务进行微调的，至少要告诉他什么是相似的，什么是不相似的，rank一下。

然后就找到了少样本的意图识别 few-shot-learning。给定support set，输入 query，输出类别。

模型里面的 距离度量，句子embedding很有用，可以用来语义匹配

few-shot-learning 大多以metric based为主，一下网络感觉都差不多，可改的的地址主要还是： **模型生成句向量**、**类向量的生成**、**距离度量计算**。


# siamese net

网络可以参考项目[siamese net](https://github.com/dhwajraj/deep-siamese-text-similarity)

要想借助bert的pre-trained model，可以按要求改一改句向量生成部分即可；
其实也就是一个句子对的分类任务，相同类别的句子对打1，不同打0，（可以使用lcqmc通用的训练微调，在这个基础上微调）

这个我没有做！

# match net

网络可以参考项目[match net](https://github.com/AntreasAntoniou/MatchingNetworks)

部分效果演示
```
loss :0.01907363533973694, acc :0.9166666666666666
loss :0.04642282426357269, acc :0.7777777777777778
loss :0.041034165769815445, acc :0.8055555555555556
dev result report:
save model:	2.390428	>2.157491
step :29001, lr:8.785421414359007e-06, loss :0.00032472479506395757, acc :1.0
```

# proto net

网络可以参考项目[proto net](https://github.com/abdulfatir/prototypical-networks-tensorflow)


部分效果演示
```
loss :16.920154571533203, acc :0.4444444444444444
loss :36.549678802490234, acc :0.3611111111111111
loss :25.368083953857422, acc :0.5
loss :15.592615127563477, acc :0.5277777777777778
loss :22.367006301879883, acc :0.3888888888888889
dev result report:
save model:	2641.328871	>2574.400064
step :4001, lr:7.999999979801942e-06, loss :0.7418746948242188, acc :0.9166666666666666
step :4002, lr:8.00199995865114e-06, loss :2.3564276695251465, acc :0.6944444444444444
```


# 使用

环境：

tf 1.14

准备 train_data.json、dev_data.json；

key 是类标志

value 是类下面的样本
```json
{
  "1": [
    "你是机器人",
    "机器人",
    "你是人还是机器人",
    "你是robot？"
  ],
  "2": [
    "我走了",
    "没事了",
    "你好，再见",
    "bye",
    "88",
    "拜拜"
  ]
}
```
运行（有exit，有数据时，更换，去掉即可）
python3 few_shot_learning_proto.py

python3 few_shot_learning_match.py