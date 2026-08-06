We observe that TP generally takes more time than EP to execute. This is because:

In TP, since each process holds a portion of every expert, input tensors need to be split and distributed across all processes. Similarly, intermediate results from sharded experts need to be gathered back together. These operations involve frequent data transfers between processes.

EP avoids this complexity because each process hosts one expert locally. Inputs routed to a specific expert are processed entirely within the corresponding process, minimizing inter-process data movement.

EP operates with each process handling one expert entirely. This simplicity translates into faster execution times for workloads where communication overhead dominates. If the number of experts matches or is close to the number of processes available, EP ensures efficient utilization of resources without introducing unnecessary complexity.

In general, if workload involves fewer experts or does not require fine-grained parallelism, EP will generally outperform TP because it minimizes inter-process communication and leverages local computation.