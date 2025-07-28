# Qwen3-30B-A3B-Mamba2权重变换及Megatron上训练部署



## 1. Transformer to Hybrid Mamba



https://github.com/jingshenghang/M1/tree/Qwen3-30B-A3B-Mamba2

Qwen3-30B-A3B-Mamba2分支，执行下面命令

```
bash transformer_to_mamba2.sh
```

关键参数

| 参数名称    | 含义                                    | 取值                                                         |
| ----------- | --------------------------------------- | ------------------------------------------------------------ |
| expand      | d_inner与hidden_size之间的扩展关系      | 2                                                            |
| attn_layers | 保留attention layer的层号（首尾需保留） | "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47" |
| n_groups    | head数量                                | 32                                                           |
| num_kv_head | kv head的数量                           | 4                                                            |
| head_dim    | head维度（≠hidden_size / num_head）     | 128                                                          |
| x_db        | k v 的维度 = num_kv_head * head_dim     | 4 * 128 = 512                                                |



## 2. convert hf to mg

使用MindSpeed-LLM的master分支中的convert_ckpt_v2.py，添加Mamba部分权重的改造，再进行转换

https://gitee.com/flippyhang/MindSpeed-LLM/tree/master_mamba_convert_ckpt/

参考PR

https://gitee.com/flippyhang/MindSpeed-LLM/pulls/2

master_mamba_convert_ckpt分支

执行下面命令：

```
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 4 \
    --load-dir /home/ascend-vllm/model/Qwen3-30B-A3B-Mamba2 \
    --save-dir /home/ascend-vllm/model/Qwen3-30B-A3B-Mamba2-tp1-pp4-ep4 \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe
```

关键参数

```
            mamba_d_inner = 4096
            mamba2_n_groups = 4
            mamba_d_state = 128
            mamba2_n_heads = 32
```



## 3. 运行Mamba Hybrid模型

https://gitee.com/flippyhang/MindSpeed-LLM/tree/2.1.0_mamba/

使用2.1.0_mamba分支

参考PR：

https://gitee.com/flippyhang/MindSpeed-LLM/pulls/1

执行下面命令：

```
bash examples/mcore/qwen3_moe/pretrain_qwen3_30b_a3b_4K_ptd_mamba.sh
```

关键参数

```
NUM_LAYERS=96
LAYER_PATTEN="*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-M-*-*-"
MAMBA_ARGS="
    --reuse-fp32-param \
    --no-shared-storage \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --num-layers ${NUM_LAYERS} \
    --mamba-ngroups 4 \
    --mamba-chunk-size 128 \
    --mamba-d-state 128 \
    --mamba-d-conv 4 \
    --mamba-expand 2 \
    --mamba-headdim 128 \
    --tokenizer-model ${TOKENIZER_PATH} \
    --hybrid-attention-ratio 0.26 \
    --hybrid-mlp-ratio 0.5 \
    --hybrid-override-pattern $LAYER_PATTEN \
    --untie-embeddings-and-output-weights \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --norm-epsilon 1e-6 \
```


