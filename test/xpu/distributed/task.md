我们需要在symm_buffer里面完成以下事宜：
1）增加allgather + permute的实现，可以支持FP8 data type

2）unpermute + Reducescatter，是只支持16bit （FP16或者BF16）

3）allgather + permute，可以支持传入scale，这个scale是per token的，shape是[tokens_per_device],所以在allgather_local_permute_fusion里面的notify_dispatch_v2，也需要收集scale。
