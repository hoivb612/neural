e:\Xbox-B612\src\neural>git clone https://github.com/intel/neural-speed.git .
Cloning into '.'...
remote: Enumerating objects: 16668, done.
remote: Counting objects: 100% (6338/6338), done.
remote: Compressing objects: 100% (1924/1924), done.
remote: Total 16668 (delta 5242), reused 4808 (delta 4413), pack-reused 10330
Receiving objects: 100% (16668/16668), 15.49 MiB | 21.17 MiB/s, done.
Resolving deltas: 100% (11577/11577), done.

main_run.obj
common.obj

..\..\LIB\RELEASE\PHI3.LIB 
  phi3.cpp
  phi3_utils.cpp
  arg_parse.cpp
  model_utils.cpp
  pool.cpp
  quant_utils.cpp
  scheduler.cpp
  util.cpp
  phi3.vcxproj -> E:\Xbox-B612\src\neural\build\lib\Release\phi3.lib

..\..\LIB\RELEASE\NE_LAYERS.LIB 
  ne_layers.c
  argsort.cpp
  bestla_gemm.cpp
  conv.cpp
  inner_product.cpp
  ip_fusion_ffn.cpp
  ip_fusion_qkv.cpp
  memory.cpp
  mha_dense.cpp
  ne_bestla.cpp
  ne_bestla_sycl.cpp


..\..\LIB\RELEASE\NE_VEC.LIB 
 ele_reduce.cpp
  ne_vec.vcxproj -> E:\Xbox-B612\src\neural\build\lib\Release\ne_vec.lib

..\..\LIB\RELEASE\CPU_VEC.LIB
  vec_arithmetic.cpp
  vec_compare.cpp
  vec_convert.cpp
  vec_set.cpp
  vec_store.cpp
  vec_load.cpp
  Generating Code...


  cpu_vec.vcxproj -> E:\Xbox-B612\src\neural\build\lib\Release\cpu_vec.lib
quant_phi3.exe --model_file e:\Xbox-B612\models\SLM\Phi-3\Phi-3-mini-4k-instruct-fp32.gguf --out_file .\itex-phi3-int4.bin --weight_dtype int4 --use_ggml 

e:\Xbox-B612\src\neural\build>bin\release\run_phi3.exe --model-name phi3 -m itex-phi3-int4.bin -s 42 -c 512 -t 12 -i --memory-f16 --color -p "Where is Paris"
Welcome to use the phi3 on the ITREX!
main: seed  = 42
AVX:1 AVX2:1 AVX512F:1 AVX512BW:1 AVX_VNNI:0 AVX512_VNNI:1 AMX_INT8:0 AMX_BF16:0 AVX512_BF16:0 AVX512_FP16:0
model_file_loader: loading model from itex-phi3-int4.bin
Loading the bin file with NE format...
load_ne_hparams  0.hparams.n_vocab = 32064
load_ne_hparams  1.hparams.n_embd = 3072
load_ne_hparams  2.hparams.n_mult = 256
load_ne_hparams  3.hparams.n_head = 32
load_ne_hparams  4.hparams.n_head_kv = 32
load_ne_hparams  5.hparams.n_layer = 32
load_ne_hparams  6.hparams.n_rot = 96
load_ne_hparams  7.hparams.ftype = 1
load_ne_hparams  8.hparams.max_seq_len = 4096
load_ne_hparams  9.hparams.alibi_bias_max = 0.000
load_ne_hparams  10.hparams.clip_qkv = 0.000
load_ne_hparams  11.hparams.par_res = 1
load_ne_hparams  12.hparams.word_embed_proj_dim = 0
load_ne_hparams  13.hparams.do_layer_norm_before = 0
load_ne_hparams  14.hparams.multi_query_group_num = 0
load_ne_hparams  15.hparams.ffn_hidden_size = 8192
load_ne_hparams  16.hparams.inner_hidden_size = 0
load_ne_hparams  17.hparams.n_experts = 0
load_ne_hparams  18.hparams.n_experts_used = 0
load_ne_hparams  19.hparams.n_embd_head_k = 0
load_ne_hparams  20.hparams.norm_eps = 0.000010
load_ne_hparams  21.hparams.freq_base = 10000.000
load_ne_hparams  22.hparams.freq_scale = 1.000
load_ne_hparams  23.hparams.rope_scaling_factor = 0.000
load_ne_hparams  24.hparams.original_max_position_embeddings = 0
load_ne_hparams  25.hparams.use_yarn = 0
load_ne_vocab    26.vocab.bos_token_id = 1
load_ne_vocab    27.vocab.eos_token_id = 32000
load_ne_vocab    28.vocab.pad_token_id = -1
load_ne_vocab    29.vocab.sep_token_id = -1
init: n_vocab                  = 32064
init: n_embd                   = 3072
init: n_head                   = 32
init: n_layer                  = 32
init: n_ff                     = 12288
init: n_parts                  = 1
init: n_embd      = 3072
init: max_seq_len      = 4096
load: ctx size   = 2696.37 MB
load: scratch0   =  512.00 MB
load: scratch1   =  512.00 MB
load: scratch2   = 1024.00 MB
load: mem required  = 4744.37 MB (+ memory per state)
...........................................................................
model_init_from_file: support_bestla_kv = 1
model_init_from_file: cpu kv self size =  192.00 MB

system_info: n_threads = 12 / 24 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | F16C = 1 | BLAS = 0 | SSE3 = 1 | VSX = 0 |
main: interactive mode on.
sampling: repeat_last_n = 64, repeat_penalty = 1.100000, presence_penalty = 0.000000, frequency_penalty = 0.000000, top_k = 40, tfs_z = 1.000000, top_p = 0.950000, typical_p = 1.000000, temp = 0.800000, mirostat = 0, mirostat_lr = 0.100000, mirostat_ent = 5.000000
generate: n_ctx = 512, tokens_length = 6, n_batch = 512, n_predict = -1, n_keep = 0


== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to LLaMa.
 - To return control without starting a new line, end your input with '/'.
 - If you want to submit another line, end your input with '\'.

Where<0x20>is<0x20>Paris▁and▁the▁Louvre

model_print_timings:        load time =  3663.84 ms
model_print_timings:      sample time =     1.61 ms /     4 runs   (    0.40 ms per token)
model_print_timings: prompt eval time =  1124.39 ms /     6 tokens (  187.40 ms per token)
model_print_timings:        eval time =  4566.98 ms /     3 runs   ( 1522.33 ms per token)
model_print_timings:       total time =  8980.38 ms
========== eval time log of each prediction ==========
prediction   0, time: 1124.39ms
prediction   1, time: 848.33ms
prediction   2, time: 2186.66ms
prediction   3, time: 1531.99ms

e:\Xbox-B612\src\neural\build>