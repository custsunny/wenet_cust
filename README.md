# wenet_cust

Modified by wenet-2.0.0，Mainly based on the deployment of libtorch

### docker environment construction  (libtorch runtime)

1. **	Create and start a linux docker mirror image(Can be modified according to their actual situation)**

   Take ubuntu : 20.04 as an example.

   ```shell
   docker pull ubuntu:20.04
   ```

   ```shell
   docker run -itd --gpus all --shm-size 4G --workdir /root -v /home/xxx/mydocker/wenet_cust1:/root/mnt --entrypoint /bin/bash --name wenet_cust1 ubuntu:20.04
   ```

   ```shell
   docker exec -it --user root wenet_cust1 /bin/bash
   ```

2. **Install the required software package ( You can change source by yourself)**

   ```shell
   apt update
   ```

   ```shell
   apt upgrade
   ```

   ```shell
   apt install -y git cmake wget build-essential vim
   ```

   Install miniconda，then creating the activation environment`pip install -r [wenet root directory ]/requirements.txt`

   set up vim

   ```shell
   vim /etc/vim/vimrc
   ```

   Add the following and save the exit

   ```shell
   set fileencodings=utf-8,ucs-bom,gb18030,gbk,gb2312,cp936
   set termencoding=utf-8
   set encoding=utf-8
   ```

 Add the following content at the end of the user configuration file ~ /.bashrc to support Chinese

   ```shell
   export LANG="C.UTF-8"
   export LANGUAGE="C.UTF-8"
   export LC_ALL="C.UTF-8"
   ```

3. **Get wenet code**

   Current directory：/root/mnt

   Decompression wenet_cust-2.2.tar.gz

   ```shell
   tar -zxvf wenet_cust-2.2.tar.gz
   ```

4. **Install libtorch environment (cmake)**

   Please ensure that cmake version is greater than or equal to 3.14

   ```shell
   cd [wenet root directory]/runtime/libtorch
   ```

   ```shell
   mkdir build && cd build
   ```

   - **cpu**

   ```shell
   cmake .. -DCMAKE_BUILD_TYPE=Release -DFST_HAVE_BIN=ON
   ```

   ```shell
   cmake --build . --config Release
   ```

   - **gpu**

   **First install cuda and cudnn, and add environment variables, a brief description of the specific steps**

   Then notice[wenet root directory]**torch version**、**cuda version**in/runtime/core/cmake/libtorch.cmake、download address and**hash value**，can be modified by yourself

   ```
   wenet_cust-2.2 correspondence torch、libtorch、cuda、cudnn version：
   torch 1.9.0+cu113 - 1.11.0+cu113
   libtorch 1.11.0+cu113
   cuda 11.3
   cudnn 8.2.1
   ```

   ```shell
   cmake .. -DCMAKE_BUILD_TYPE=Release -DFST_HAVE_BIN=ON -DGPU=ON
   ```

   ```shell
   cmake --build . --config Release
   ```

  Note : If you can not download, you can download wget first, and then move to the target directory.

5. **File preparation**

   Folders with the following directory and file structure can be created locally：

   ```
   work
   --make_graph
   ----lm
   ------lm.arpa
   ----dict
   ------lexicon.txt
   ------units.txt
   
   --model_dir
   ----final.pt
   ----final.zip
   ----global_cmvn
   ----train.yaml
   ----units.txt
   
   --wav_dir
   ----test1.wav
   ----test2.wav
   ----text
   ----wav.scp
   
   --path.sh
   ```

   **Filespec：**

   - [ ] **`make_graph`**

   ```
   --make_graph
   ----lm
   ------lm.arpa
   ----dict
   ------lexicon.txt
   ------units.txt
   ```

   `make_graph`stores files for composition，for**adding language models**，includes language models`make_graph/lm/lm.apra`、Word to character file`make_graph/dict/lexicon.txt`and modeling unit file`make_graph/dict/units.txt`

   - `lm.apra`

     `lm.apra` is an n-gram language model about words or characters， Its name must be lm.arpa

   - `units.txt`

     `units.txt`is the modeling unit，that is, data / dict / lang _ char.txt generated by wenet data preparation，Examples of its contents are as follows：

     ```
     <blank> 0
     <unk> 1
     ▁ 2
     A 3
     B 4
     C 5
     D 6
     E 7
     F 8
     G 9
     H 10
     I 11
     J 12
     K 13
     L 14
     M 15
     N 16
     O 17
     P 18
     Q 19
     R 20
     S 21
     T 22
     U 23
     V 24
     W 25
     X 26
     Y 27
     Z 28
     ○ 29
     一 30
     丁 31
     七 32
     万 33
     丈 34
     三 35
     ```

    The first column is modeling unit，Chinese end-to-end speech recognition generally uses words as modeling units，The second column is the modeling unit serial number。`<blank>`represents empty,，`<unk>`represents a word that is not in the list，The last one`<sos/eos>`represents the beginning or end，The rest are ordinary words or characters。

   - `lexicon.txt`

     `lexicon.txt`is a file from word to character，**When**`lm.apra`**is a word-based language model**，the first column of' lexicon.txt' is all words，The second to the last column are the characters that correspond to these words，The examples are as follows：

     ```
     啊 啊
     啊啊啊 啊 啊 啊
     阿 阿
     阿尔 阿 尔
     阿根廷 阿 根 廷
     阿九 阿 九
     阿克 阿 克
     阿拉伯数字 阿 拉 伯 数 字
     阿拉法特 阿 拉 法 特
     阿拉木图 阿 拉 木 图
     阿婆 阿 婆
     阿文 阿 文
     阿亚 阿 亚
     阿育王 阿 育 王
     阿扎尔 阿 扎 尔
     ```

     **When**`lm.apra`**is a word-based language model**，characters can be regarded as words，use all the words in `units.txt` as the contents of the first and second columns in `lexicon.txt`(except for`<blank>`、`<unk>`、`<sos/eos>`)，The example of `lexicon.txt`is as follows：

     ```
     乒 乒
     乓 乓
     乔 乔
     乖 乖
     乘 乘
     乙 乙
     乜 乜
     九 九
     乞 乞
     也 也
     ```

     Using the following command from`units.txt`generate`lexicon.txt`

     ```shell
     # 21066is the number of modeling units-1，It must be modified by itself
     sed -n '3,21066p' units.txt | awk -F ' ' '{print $1,$1}' > lexicon.txt
     ```

   - [ ] **`model_dir`**

     ```
     --model_dir
     ----final.pt
     ----final.zip
     ----global_cmvn
     ----train.yaml
     ----units.txt
     ```

     `model_dir`stores model-related files，Including the checkpoint file final.pt generated after model training and parameter averaging.`final.pt`，libtorch model final.zip exported by torch script`final.zip`，global cmvn file`global_cmvn`，training configuration file `train.yaml`(Under the folder generated by each example training，as`exp/conformer/train.yaml`)，modeling unit file`units.txt`(It is consistent with 'data / dict / lang _ char.txt' after data preparation of each example)。

    ' Final.zip' and 'units.txt' are required files for speech recognition in libtorch runtime environment，`final.pt`、`global_cmvn`、`train.yaml`and`units.txt`are required files for further training / parameter fine-tuning of the model.。

   - [ ] **`wav_dir`**

     ```
     --wav_dir
     ----test1.wav
     ----test2.wav
     ----text
     ----wav.scp
     ```

    The audio files to be tested are stored under wav _ dir，The number of audio channels is 1，The sampling rate is 16000，The sampling size is 2，The audio format is wav，In addition to audio files，it also includes`tet`and`wav.scp`，

     `text`is label file，The first item of each line is audio id，The second item is the text corresponding to the audio，The middle is separated by spaces，Examples are as follows：

     ```
     001 甚至出现交易几乎停滞的情况
     002 一二线城市虽然也处于调整中
     ```

     `wav.scp`is audio path file，The first item of each line is audio id，The second item is the audio path，The middle is separated by spaces. The examples are as follows：

     ```
     001 /root/mnt/work/wav_dir/test1.wav
     002 /root/mnt/work/wav_dir/test2.wav
     ```

   - [ ] **`path.sh`**

     ```
     --path.sh
     ```

     `path.sh`is an environment variable file，For speech recognition in libtorch runtime environment，Where ' WENET _ DIR ' needs to point to the wenet root directory，The details are as follows：

     ```shell
     export WENET_DIR=/root/mnt/wenet_cust-2.2
     export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
     export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
     export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH
     ```

6. **Put the work folder that contains the above into the docker container，directory is/root/mnt/work）**

   Put ' tools ' and ' mytools ' under ' wenet _ cust-2.2 ' in the ' / root / mnt / work ' directory
   ```shell
   cd /root/mnt/work
   ```

   ```shell
   cp -r /root/mnt/wenet_cust-2.2/tools .
   ```

   ```shell
   cp -r /root/mnt/wenet_cust-2.2/mytools .
   ```

   Give authority

   ```shell
   chmod 775 -R /root/mnt
   ```

   Convert a group and a person

   ```shell
   chgrp root -R /root/mnt
   chown root -R /root/mnt
   ```

7. **Recognition without language model[Non-streaming]**

   Activate environment variables

   ```shell
   cd /root/mnt/work
   . ./path.sh
   ```

   Identify a single audio

   ```shell
   decoder_main --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --unit_path model_dir/units.txt --model_path model_dir/final.zip --wav_path wav_dir/test.wav
   ```

   - Identify multiple audios(multithread)

   ```shell
   ./tools/decode.sh --nj 4 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 wav_dir/wav.scp wav_dir/text model_dir/final.zip model_dir/units.txt result_without_lm
   ```

8. **Recognition in the case of adding language model[Non-streaming]**

   The language model of wenet is described in detail`[wenet root directory]/docs/lm/lm.md`，The following is a brief description of ' T.fst, L.fst, G.fst '

   ![TLG](E:\语音识别学习\wenet_cust\docs\images\TLG.png)

   - **Plus word-based language model**

  Use the following command，according to`units.txt`，generate`lexicon.txt`，**If there is already ' lexicon.txt ' then ignore**

   ```shell
   cd /root/mnt/work/make_graph/dict
   ```

   ```shell
   mv lexicon.txt lexicon_backup.txt
   ```

   ```shell
   # 查看units.txt有多少行
   wc units.txt
   ```

   ```shell
   # 如果units.txt有21067行
   sed -n '3,21066p' units.txt | awk -F ' ' '{print $1,$1}' > lexicon.txt
   ```

   如前文所述，此时`lexicon.txt`示例如下

   ```
   乒 乒
   乓 乓
   乔 乔
   乖 乖
   乘 乘
   乙 乙
   乜 乜
   九 九
   乞 乞
   也 也
   ```

   根据`units.txt`和`lexicon.txt`构建`T.fst`和`L.fst`

   ```shell
   cd /root/mnt/work/
   ```

   ```shell
   ./tools/fst/compile_lexicon_token_fst.sh make_graph/dict make_graph/temp make_graph/lang_temp
   ```

   `T.fst`和`L.fst`构建完成之后，使用`make_tlg.sh`构建`TLG.fst`

   ```shell
   ./tools/fst/make_tlg.sh make_graph/lm make_graph/lang_temp/ make_graph/lang
   ```

   使用`TLG.fst`进行加语言模型的语音识别

   - 识别单个音频

   ```shell
   decoder_main  --acoustic_scale 4.0 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --rescoring_weight 0.5 --beam 15.0 --lattice_beam 4.0 --max_active 7000 --blank_skip_thresh 0.98 --fst_path make_graph/lang/TLG.fst --dict_path make_graph/lang/words.txt --unit_path model_dir/units.txt --model_path model_dir/final.zip --wav_path wav_dir/test.wav
   ```

   - 识别多个音频(多线程)

   ```shell
   ./tools/decode.sh --nj 4  --acoustic_scale 4.0 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --rescoring_weight 0.5 --beam 15.0 --lattice_beam 4.0 --max_active 7000 --blank_skip_thresh 0.98 --fst_path make_graph/lang/TLG.fst --dict_path make_graph/lang/words.txt wav_dir/wav.scp wav_dir/text model_dir/final.zip model_dir/units.txt result
   ```

   `nj`is the number of threads，

   `wav_dir/wav.scp`为音频列表文件，

   `wav_dir/text`为标签文件，如果不需要计算字错率或没有标签文件，只需要指定一个存在的空文件即可。

   `model_dir/units.txt`为建模单元文件，其与`make_graph/dict/units.txt`一致。

   `make_graph/lang/TLG.fst`为构图得到的WFST图，`make_graph/lang/words.txt`为该`TLG.fst`对应的词典，是由`tools/fst/make_tlg.sh`生成的。

   `acoustic_scale`用于调整声学得分的大小，使得声学得分与语言模型等分数在同一个数量级上，`rescoring_weight`为语言模型重打分比重，即使重打分比重为0，也会因WFST图中的一些常数值而使得识别结果与不加语言模型不一样。

   - **加基于词的语言模型**

   查看准备好的`lexicon.txt`，建议使用语言模型中所有的词作为`lexicon.txt`第一列所有的词，`lexicon.txt`示例如下：

   ```
   啊 啊
   啊啊啊 啊 啊 啊
   阿 阿
   阿尔 阿 尔
   阿根廷 阿 根 廷
   阿九 阿 九
   阿克 阿 克
   阿拉伯数字 阿 拉 伯 数 字
   阿拉法特 阿 拉 法 特
   阿拉木图 阿 拉 木 图
   阿婆 阿 婆
   阿文 阿 文
   阿亚 阿 亚
   阿育王 阿 育 王
   阿扎尔 阿 扎 尔
   ```

   Use`mytools/remove_oov_in_lexicon.py`去除`lexicon.txt`中含 ”超纲字“ 的词及其对应的字，“超纲字”指`units.txt`中没有的字

   ```
   mv make_graph/dict/lexicon.txt make_graph/dict/lexicon_backup.txt
   ```

   ```shell
   python3 mytools/remove_oov_in_lexicon.py --units_path model_dir/units.txt --lexicon_path make_graph/dict/lexicon_backup.txt --new_lexicon_path make_graph/dict/lexicon.txt
   ```

   得到符合要求的`lexicon.txt`后，下面的步骤与加基于字的语言模型一样

   根据`units.txt`和`lexicon.txt`构建`T.fst`和`L.fst`

   ```shell
   cd /root/mnt/work/
   ```

   ```shell
   ./tools/fst/compile_lexicon_token_fst.sh make_graph/dict make_graph/temp make_graph/lang_temp
   ```

   `T.fst`和`L.fst`构建完成之后，使用`make_tlg.sh`构建`TLG.fst`

   ```shell
   ./tools/fst/make_tlg.sh make_graph/lm make_graph/lang_temp/ make_graph/lang
   ```

   使用`TLG.fst`进行加语言模型的语音识别

   - 识别单个音频

   ```shell
   decoder_main  --acoustic_scale 4.0 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --rescoring_weight 0.5 --beam 15.0 --lattice_beam 4.0 --max_active 7000 --blank_skip_thresh 0.98 --fst_path make_graph/lang/TLG.fst --dict_path make_graph/lang/words.txt --unit_path model_dir/units.txt --model_path model_dir/final.zip --wav_path wav_dir/test.wav
   ```

   - 识别多个音频(多线程)

   ```shell
   ./tools/decode.sh --nj 4  --acoustic_scale 4.0 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --rescoring_weight 0.5 --beam 15.0 --lattice_beam 4.0 --max_active 7000 --blank_skip_thresh 0.98 --fst_path make_graph/lang/TLG.fst --dict_path make_graph/lang/words.txt wav_dir/wav.scp wav_dir/text model_dir/final.zip model_dir/units.txt result
   ```

   

   







------

# WeNet

[**中文版**](https://github.com/wenet-e2e/wenet/blob/main/README_CN.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**Roadmap**](ROADMAP.md)
| [**Docs**](https://wenet-e2e.github.io/wenet/)
| [**Papers**](https://wenet-e2e.github.io/wenet/papers.html)
| [**Runtime (x86)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86)
| [**Runtime (android)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet)
| [**Pretrained Models**](docs/pretrained_models.md)

**We** share neural **Net** together.

The main motivation of WeNet is to close the gap between research and production end-to-end (E2E) speech recognition models,
to reduce the effort of productionizing E2E models, and to explore better E2E models for production.

Note: please read `modify_list.txt` before use, please ensure that the `wenet` folder in each example is consistent with the `wenet` in the root directory

## Highlights

* **Production first and production ready**: The core design principle of WeNet. WeNet provides full stack solutions for speech recognition.
  * *Unified solution for streaming and non-streaming ASR*: [U2 framework](https://arxiv.org/pdf/2012.05481.pdf)--develop, train, and deploy only once.
  * *Runtime solution*: built-in server [x86](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86) and on-device [android](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet) runtime solution.
  * *Model exporting solution*: built-in solution to export model to LibTorch/ONNX for inference.
  * *LM solution*: built-in production-level [LM solution](docs/lm.md).
  * *Other production solutions*: built-in contextual biasing, time stamp, endpoint, and n-best solutions.

* **Accurate**: WeNet achieves SOTA results on a lot of public speech datasets.
* **Light weight**: WeNet is easy to install, easy to use, well designed, and well documented.

## Performance Benchmark

Please see `examples/$dataset/s0/README.md` for benchmark on different speech datasets.

## Installation(Python Only)

If you just want to use WeNet as a python package for speech recognition application,
just install it by `pip`, please note python 3.6+ is required.

``` sh
pip3 install wenet
```

And please see [doc](runtime/binding/python/README.md) for usage.


## Installation(Training and Developing)

- Clone the repo

``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch=1.10.0 torchvision torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- Optionally, if you want to use x86 runtime or language model(LM),
  you have to build the runtime as follows. Otherwise, you can just ignore this step.

``` sh
# runtime build requires cmake 3.14 or above
cd runtime/server/x86
mkdir build && cd build && cmake .. && cmake --build .
```

## Discussion & Communication

Please visit [Discussions](https://github.com/wenet-e2e/wenet/discussions) for further discussion.

For Chinese users, you can aslo scan the QR code on the left to follow our offical account of WeNet.
We created a WeChat group for better discussion and quicker response.
Please scan the personal QR code on the right, and the guy is responsible for inviting you to the chat group.

If you can not access the QR image, please access it on [gitee](https://gitee.com/robin1001/qr/tree/master).

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/robin1001/qr/blob/master/binbin.jpeg" width="250px"> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

Or you can directly discuss on [Github Issues](https://github.com/wenet-e2e/wenet/issues).

## Contributors

| <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="250px"></a> | <a href="http://lxie.npu-aslp.org" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/nwpu.png" width="250px"></a> | <a href="http://www.aishelltech.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/aishelltech.png" width="250px"></a> | <a href="http://www.ximalaya.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/ximalaya.png" width="250px"></a> | <a href="https://www.jd.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/jd.jpeg" width="250px"></a> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <a href="https://horizon.ai" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/hobot.png" width="250px"></a> | <a href="https://thuhcsi.github.io" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/thu.png" width="250px"></a> | <a href="https://www.nvidia.com/en-us" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/nvidia.png" width="250px"></a> |                                                              |                                                              |

## Acknowledge

1. We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet) for transformer based modeling.
2. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for WFST based decoding for LM integration.
3. We referred [EESEN](https://github.com/srvk/eesen) for building TLG based graph for LM integration.
4. We referred to [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer/) for python batch inference of e2e models.

## Citations

``` bibtex
@inproceedings{yao2021wenet,
  title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
  author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
  booktitle={Proc. Interspeech},
  year={2021},
  address={Brno, Czech Republic },
  organization={IEEE}
}

@article{zhang2022wenet,
  title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
  author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
  journal={arXiv preprint arXiv:2203.15455},
  year={2022}
}
```
