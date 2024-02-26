# [AP4064 Weather and Artificial Intelligence II] 23 Spring HW02: CIFAR-10 with CNN (1chooo)

> AP4063 - Weather and Artificial Intelligence ⅠⅠ Homework 2
> 
> Year: 2023 Spring   
> Lecturer: Che-Wei Chou (周哲維)   
> Student: Hugo ChunHo Lin (林群賀)  

Please print all the interactive output without resorting to print, not only
the last result. To do this, you should add the following code in Jupyter
cell to the beginning of the file.

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

### 一、

Cifar10 資料即包含 6 萬筆 32*32 低解析度之彩色圖片，其中 5 萬筆為訓練集；1 萬筆為測試集。所有圖片被分為 10 個類別：

| label |   Type   | label |    Type    |
| :---: | :------: | :---: | :--------: |
|   0   | airplane |   1   | automobile |
|   2   |   bird   |   3   |    cat     |
|   4   |   deer   |   5   |    dog     |
|   6   |   frog   |   7   |   horse    |
|   8   |   ship   |   9   |   truck    |

請在 Keras 中利用 CNN 盡你所能的訓練模型並完整報告出最好的一次結果及與第一次作業您訓練出來的 DNN 模型做比較。

> [!WARNING]
> 禁止使用 pre-trained model 課堂上未提及的方法請附加說明

```python
import Keras
from Keras.datasets import cifar10
```


### 二、

承題一，使用 pre-trained model VGG16 作為模型，並搭配 data augmentation (可fine-tuning)，將預測結果與題一做比較。使用 `VGG16(weights=’imagenet’,include_top=False,input_shape=(64,64,3))`
