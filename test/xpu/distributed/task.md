考虑基于ishmem做一个allgather+local permute fusion，具体是：
1）GPU到GPU的通信必须跨RDMA NIC，此时是无法直接访问remote device的memory地址的，只能通过ishmem的API。
2）实现一个基于ishmem的allgather_permute native kernel，但是需要考虑跨NIC的情况，例如无法直接读写remote的API，只能通过ishmem的API
3）阅读我本地的ishmem的code /root/cherry/ep_ws/ishmem_ibgda ，并告诉我要完成这个kernel，需要调用ishmem的什么样的API。
4）代码实现之前，请先跟我讨论实现的design
5) 所有的API实现必须配套UT来验证accuracy
