# HandWritingRec
> model for recognizing handwriting

+ deconstruct repository from PaddlePaddle 

## Coding Records

### 20231026
1. add config.yml & program.py(to be revised)
2. add DataLoader.py(need to be checked by chatgpt)
3. add simple_dataset.py




## Todo
1. after finishing coding,adjust the config.yml


## Knowledge Records
+ 在ppocr.data存放了很多dataset.py 是不同数据集的加载方式吗，默认使用simple_dataset? 和我理解的一样
+ Python 没有语言原生的可见性控制，而是靠一套需要大家自觉遵守的”约定“下工作。比如下划线开头的应该对外部不可见。同样，__all__ 也是对于模块公开接口的一种约定，比起下划线，__all__ 提供了暴露接口用的”白名单“。一些不以下划线开头的变量（比如从其他地方 import 到当前模块的成员）可以同样被排除出去。控制 from xxx import * 的行为
  + 代码中当然是不提倡用 from xxx import * 的写法的，但是在 console 调试的时候图个方便还是很常见的。如果一个模块 spam 没有定义 __all__，执行 from spam import * 的时候会将 spam 中非下划线开头的成员都导入当前命名空间中，这样当然就有可能弄脏当前命名空间。如果显式声明了 __all__，import * 就只会导入 __all__ 列出的成员。如果 __all__ 定义有误，列出的成员不存在，还会明确地抛出异常，而不是默默忽略。


## Reference
+ [手写文字识别](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/applications/%E6%89%8B%E5%86%99%E6%96%87%E5%AD%97%E8%AF%86%E5%88%AB.md)
