## Test configuration (model inference)
- 2 instances, functionality only(TBD?)
- Model: LLAMA3.1 70B, LLAMA4 17B
- Parallelism
  - TP: allreduce, allgather
    - allreduce activation shape (input shape): [BS * sequence_length, hidden_size]
    - allgather shape(logits shape): [BS * sequence_length, hidden_size] (???)
  - EP: allgatherv, reducescatterv
    - allgatherev shape (?)
    - reducescatterv shape (?)
  - PP: no need(?)
- Data type:
  - Collective w/ reduction: BF16
  - Collectives w/o reducetion: BF16, MXFP8

## LLAMA3.1 70B  recipe
- https://huggingface.co/meta-llama/Llama-3.1-70B/main/config.json


## LLAMA4  17B  recipe
- https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/blob/main/config.json
- https://github.com/intel-innersource/frameworks.ai.pytorch.gpu-models/blob/master/presi-models/reduced-llama4/configs/Llama-4-Maverick-17B-128E-Instruct.json 

