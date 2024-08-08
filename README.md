# GPT-SoVITS 自用小工具
## 1. 项目简介
本项目是为GPT-SoVITS提供的一个简易webui，功能包括选择参考音频、手动修改音素。原项目webui在使用时较为繁杂，难以更换参考音频，无法手动修改音素。故本项目对原项目进行了简化，使得更易于使用。 该Webui改造自[SayaSS](https://github.com/SayaSS)大哥的webui。

## 2. 使用方法
将本项目所有文件放入GPT-SoVITS项目根目录下，然后按照以下步骤进行操作：

- 在models目录下放置模型文件夹，文件夹内包括transcript.txt(训练时使用的标注list)，reference_audio文件夹(标注list所包含的所有音频)
- 在models/models_info.json中填写模型信息，信息结构如下：

```
{
    "sample":{ <----模型名称
    "transcript_path": "models/sample/transcript.txt", <----标注路径
    "gpt_weight": "models/sample/sample-e00.ckpt",  <----GPT模型路径
    "sovits_weight": "models/sample/sample_e0_s0.pth", <----SoVITS模型路径
    "title": "test", <----标题
    "cover": "", <----封面图片路径（可选）
    "example_reference": "example_reference"  <----初始载入时默认的参考音频的文本（如不填写/填写错误，则默认为第一条）        
    }   
}
```
- 直接python app.py运行即可，环境需求与GPT-SoVITS项目保持一致。

## 3. 更新日语词典
pyopenjtalk词典较为陈旧，但提供了更新词典的方法。在项目根目录下运行以下命令即可更新词典：
```
python compile.py
```
该指令会将本项目根目录下的userdic.csv编译为pyopenjtalk所需的词典文件，在app.py执行时会自动载入。
该词典目前包含tdmelodic的词条，以及其中英语有关的词条。可以自己按需添加新词条。