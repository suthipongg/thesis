cedar: Composable and Optimized Machine Learning Input Data Pipelines

arXiv:2401.08895v1 [cs.LG] 17 Jan 2024

Mark Zhao
Stanford University

Emanuel Adamiak
Stanford University

Abstract

jobs due to model/hyperparameter exploration [36, 82] and
random augmentations [16, 58]. For example, a recommendation pipeline may hash categorical features based on the
dimensions of the model itself [69], while a computer vision
pipeline may distort each image randomly every time it is ingested [16]. Thus, materializing all possible transformations
a priori is cost-prohibitive if not infeasible.
Input data pipelines present unique requirements such as
domain-specific tensor operations [65, 72], ML framework
integrations [1,59], sub-ms processing latencies [7], and rightsizing to match accelerator demands [48]. Thus, traditional
stream processing frameworks (e.g., Spark Streaming [79]
and Flink [12]) are ill-suited for input data pipelines. Instead,
ML practitioners rely on input data systems, such as PyTorch’s
DataLoader [64] and TensorFlow’s tf.data [52], to continuously extract training data from storage, apply complex transformations to each sample, and load processed mini-batches
onto accelerators throughout the training job’s lifetime.
Recently, the input data throughput demanded by training
jobs has grown at an immense rate [48, 52, 81, 82], driven
by specialized hardware [8, 35, 68], optimized software techniques [29, 30], and massive training clusters [42, 47, 50]. To
avoid input data bottlenecks which cripple training throughput [48, 82], companies such as Google [7], Alibaba [81], and
Meta [82] have deployed dedicated distributed input data services. Unfortunately, simply scaling-out input data pipelines
is resource-intensive and inefficient. Meta’s DPP can require
dozens of CPU servers to support a single GPU server [82],
while a single model at Google can require more than five
thousand preprocessing workers [7]! Optimizing both the
performance and efficiency of input data systems is essential.
To understand the impacts and complexities of optimizing
these systems, we begin by presenting a study of many existing and novel optimization techniques. For example, offloading computations [28,39,52,75,81,82] (e.g., to disaggregated
workers) and prefetching [52] allow input data systems to
parallelize and overlap input data processing with training.
Caching intermediate input data [15, 28, 34, 37, 40, 48, 80, 83]
exploits reuse between training epochs, while fusing opera-

The input data pipeline is an essential component of each machine learning (ML) training job. It is responsible for reading
massive amounts of training data, processing batches of samples using complex of transformations, and loading them onto
training nodes at low latency and high throughput. Performant
input data systems are becoming increasingly critical, driven
by skyrocketing data volumes and training throughput demands. Unfortunately, current input data systems cannot fully
leverage key performance optimizations, resulting in hugely
inefficient infrastructures that require significant resources –
or worse – underutilize expensive accelerators.
To address these demands, we present cedar, a programming model and framework that allows users to easily build,
optimize, and execute input data pipelines. cedar presents an
easy-to-use programming interface, allowing users to define
input data pipelines using composable operators that support
arbitrary ML frameworks and libraries. Meanwhile, cedar
transparently applies a complex and extensible set of optimization techniques (e.g., offloading, caching, prefetching,
fusion, and reordering). It then orchestrates processing across
a customizable set of local and distributed compute resources
in order to maximize processing performance and efficiency,
all without user input. On average across six diverse input data
pipelines, cedar achieves a 2.49×, 1.87×, 2.18×, and 2.74×
higher performance compared to tf.data, tf.data service, Ray
Data, and PyTorch’s DataLoader, respectively.

1

Christos Kozyrakis
Stanford University

Introduction

Every deep machine learning (ML) training job relies on systems that process raw data into prepared training samples.
While some preprocessing operations may be performed “offline” by traditional parallel processing frameworks such as
Spark [78] or Beam [2], each training job also relies on an input data pipeline to further process each sample “online” as it
is ingested by the model. This is because training jobs rely on
diverse transformations that can vary both across and within
1

Raw Data
Sources

tors [52] reduces data transfer overheads. We also analyze
how reordering operators that affect sample sizes can significantly reduce work on a per-sample basis.
While numerous impactful optimization techniques exist,
current systems fail to fully capitalize on their benefits because they are: a) Fragmented. Optimization techniques are
dispersed and deeply integrated across myriad input data systems [7, 52, 63, 67, 81, 82], forcing users to choose amongst
optimizations. b) Specialized. Meanwhile, due to the immense
diversity of ML ecosystems, many systems are limited in their
ability to optimize general workloads. For example, tf.data service [7] requires TensorFlow graphs [71], precluding the use
of many third-party domain-specific libraries, Python UDFs,
and ML frameworks such as PyTorch. c) Inextensible. As we
show in Section 3, leveraging new optimizations and backends (e.g., disaggregated workers) presents a complex search
space that current systems cannot reason about. In short, there
lacks a framework that supports general input data pipelines,
applies and combines impactful-yet-complex optimization
techniques, and utilizes diverse ML infrastructures.
To address these limitations, this paper presents cedar, a
composable and optimized programming model and framework for ML input data pipelines. cedar provides ML practitioners an API to define general input data pipelines and a
runtime that automatically applies complex optimizations and
orchestrates processing across an extensible set of execution
backends. cedar aims to maximize the performance of input
data processing in order to address costly bottlenecks and
inefficiencies in current input data systems [7, 48, 82]. It does
so by exposing three composable and extensible interfaces.
a) Programming interface. cedar allows ML practitioners
to easily define stateless input data pipelines (Features) by
chaining together modular native and higher-order operators
(Pipes) functionally. cedar is Python-native and is agnostic
to users’ choice of ML framework or libraries. It also allows
users to specify a set of minimal-yet-expressive hints that
permit cedar to apply optimizations while maintaining the
semantic correctness of black-box UDFs. b) Optimization
interface. cedar models each Feature as a logical dataflow
graph; cedar then exposes well-defined methods to monitor
and manage the Feature’s execution. cedar’s Optimizer
leverages this interface to statically optimize and dynamically tune processing using an extensible set of optimization
techniques. c) Execution interface. Finally, cedar does not
aim to replace the myriad existing input data processing systems [21, 52, 53, 60, 64, 67]. Instead, it offers an interface
that allows the Optimizer to offload Pipes to an extensible
set of local and distributed processing backends. This allows
cedar to balance distinct strengths and weaknesses between
backends for each pipeline, choosing the best tool for the job.
We implemented both local (multiprocessing) and distributed (Ray [49]) backends, and we evaluated cedar on
a diverse set of six input data pipelines across ML domains.
By understanding the complex systems dynamics that arise

Training Jobs
Offline Feature
Engineering

Accelerator

ML
Datasets

Input Data
Pipeline

Accelerator
Input Data
Pipeline

Figure 1: ML training data pipelines consist of an offline
feature engineering and an online input data processing stage.
in each pipeline, cedar successfully leverages a mix of optimizations and backends to maximize each pipeline’s performance. On average, cedar outperforms state-of-the-art
ML input data systems, including tf.data [52], tf.data service [7], Ray Data [67], and PyTorch’s DataLoader [64] by
2.49×, 1.87×, 2.18×, and 2.74×, respectively. We plan to
open source cedar upon publication, providing a platform
for the community to further develop and leverage input data
optimizations.

2

Background and Motivation

ML data ingestion. ML training jobs inherently rely on data
ingestion pipelines to transform raw operational data into
structured samples (i.e., tensors) interpretable by ML frameworks such as PyTorch [59] and TensorFlow [1]. As shown in
Figure 1, these pipelines traditionally consist of two phases:
offline feature engineering and online input data processing.
Feature engineering pipelines validate, aggregate, join, and
transform raw operational data into structured datasets, off
the critical path of training. For example, a recommendation
dataset may require joining production databases and log
streams across services [82]. Since feature engineering predominantly requires traditional extract-transform-load (ETL)
tasks, ML practitioners commonly use general-purpose distributed processing frameworks such as Spark [78], Flink [12],
and Beam [2]. Because of the diverse scale of training workloads, dataset sizes, and ML domains, these datasets can be
stored across a wide variety of systems, ranging from training
nodes’ local filesystems to cloud data lakes [5, 17, 57, 83], as
raw, binary, or structured files [22, 24, 25, 73].
Why we need online input data processing. ML models rely
on numerous operations that vary heavily across jobs or even
across epochs (i.e., passes on the dataset) within a job. For example, NLP practitioners often experiment with tokenization
algorithms [32] across model architectures. Recommendation
models require different hashing configurations depending
on their embedding table dimensions [69]. Many domains,
such as computer vision [16] and speech [58], apply random
augmentations that recompute each sample every epoch to
improve model generalization. Materializing unique datasets
for each training job (or even each epoch) would be wildly inefficient. Instead, each training job uses an input data pipeline
2

Float

RandCrop

RandFlip

Jitter

Grayscale

Blur

Normalize

Batch

Image
Dataset

Decode

to perform the “last-mile” of processing by extracting training
samples from stored datasets, transforming them into readyto-use mini-batches of tensors, and loading tensors into the
host memory of the training node – all on-the-fly.
Why we need input data systems. Input data pipelines
present requirements distinct from traditional ETL tasks, demanding a dedicated class of systems. Deep learning jobs
typically train models using mini-batch gradient descent. At
each training step, the ML framework (e.g., PyTorch or TensorFlow) copies a mini-batch from host to accelerator memory,
and the accelerator performs backpropagation on the minibatch. The ML framework performs many such steps until
the model reaches convergence, potentially requiring multiple
passes (epochs) over the raw dataset.
As a result, the input data pipeline must right-size resources and generate each mini-batch with strict low latency (O(100µs − 1ms) per step) [7] and high throughput
(O(10GB/s) per node) [82] requirements throughout the lifetime of the job, which can take days to weeks. Violations of
these requirements can bottleneck expensive accelerators [48].
Furthermore, instead of joins, aggregations, and sorts on structured data, ML input data pipelines predominantly operate on
unstructured tensors within a mini-batch granularity and rely
on myriad domain-specific Python libraries and UDFs.
Why current input processing frameworks are insufficient.
ML practitioners have historically relied on first-party input
data libraries provided by ML frameworks (e.g., PyTorch’s
DataLoader [64] and TensorFlow’s tf.data [52]), which utilize
local resources (i.e., training nodes’ host CPUs) to perform
processing. Unfortunately, as throughput demands continue
to rise, these frameworks are struggling to keep up [7, 82].
Thus, recent works have begun to explore a breadth of optimization and scaling techniques such as parallelism and
disaggregation [28, 39, 52, 75, 81, 82], caching [15, 28, 34, 37,
40, 43, 48, 80, 83], operator fusion [52, 67], and inter-job coordination [28, 36, 40, 48, 80]. Furthermore, as we explore next,
traditional query optimization techniques such as reordering [38] can further yield benefits for input data pipelines.
While there are many promising optimizations, current systems fail to take full advantage of and automate them because
they are: a) Fragmented. Current optimization techniques in
input data systems are nascent and fragmented across respective frameworks, placing prohibitive restrictions on ML practitioners. For example, users wishing to adopt Plumber’s [39]
resource allocator must use tf.data. b) Specialized. ML input data pipelines rely heavily on a diverse mix of domainspecific libraries and frameworks as opposed to a standardized
interface (e.g., SQL). Thus, applying existing input data and
query optimization techniques across general workloads is
difficult, forcing systems to make restrictions. For instance,
while tf.data service can leverage TensorFlow’s graph optimizations [52], users cannot harness its benefits for PyTorch
pipelines [71]. c) Inextensible. As we explore next, extending
and combining optimization techniques presents a complex

(D)

(F)

(C)

(H)

(J)

(G)

(R)

(N)

(B)

Figure 2: The SimCLR input data pipeline applies a sequence
of augmentations to each image. Abbreviated letters in parentheses denote the operator throughout this section.

Figure 3: Pipeline execution time, normalized to the baseline,
by offloading operators across local and remote backends.
search space about which current input data systems cannot
reason. While there is an urgency to address the large and increasing resource demands of input data systems, there lacks
a means to develop, compose, and leverage key optimizations
across the wide set of input data pipelines and systems in use.

3

Input Processing Optimization Trade-offs

We begin by demonstrating the importance – and complexities
– of applying optimizations to input data pipelines. We evaluate a set of key optimization techniques – offloading, fusion,
caching, reordering, and prefetching – on a representative
computer vision input data pipeline for SimCLR [13, 14] (CV)
as shown in Figure 2. We performed experiments on an 8-core
VM on Google Cloud (n2-standard-8) with some experiments using an additional 32-core VM (n2-standard-32).
The pipeline read a subset of the ImageNet dataset [19] stored
on the filesystem of the 8-core VM. Section 6 further explores
natural language and speech pipelines.
Offloading: Choose the right backend for each operator.
Many input data systems offload the entire pipeline execution
to local and remote backends (e.g., thread/process pools [52,
64] or distributed workers [7,81,82]) to parallelize processing.
However, each backend presents trade-offs between available
resources and incurred overheads, impacting the performance
of each operator differently. To maximize overall performance,
input data systems should carefully select each operator’s
backend depending on the operator’s unique performance
characteristics and leverage both local and remote resources.
To demonstrate this, Figure 3 shows the execution time of
the CV pipeline across various backends, normalized to the
baseline which executes all operators within the main data
loading process on the 8-core VM. mp offloads execution of
the entire pipeline to a local multiprocess pool. mp+r offloads
the entire pipeline to multiple processes on the remote 32-core
VM, using the local multiprocess pool to send images and receive batched tensors via Ray. Surprisingly, more parallelism
3

Figure 4: Pipeline execution time, normalized to the baseline,
by executing fused operators (see Figure 2) remotely.

Figure 6: Execution time (without offloading) by reordering
various operators in the CV pipeline (see Figure 2).
experiment, the output of a different operator was materialized
during the first epoch. In subsequent epochs, this data was
then read from cache and processed by downstream operators.
These results demonstrate two important takeaways. First,
caching can harm pipeline throughput, e.g., caching the int8
to fp32 conversion (F). This is because operators can increase
the size of each sample; caching can incur more I/O overheads relative to compute saved. Secondly, caching after any
random operator (e.g., random crop) would unacceptably reduce the model’s accuracy (see §2). Considering these factors,
applying caching to the CV pipeline is ineffective without further optimizations! Caching prior to the first random operator
(crop - C) increases execution time by 81%, while caching
the decoded image only reduces execution time by 5% but
requires 5.86× more cache capacity than the dataset itself.
Reordering: Opportunity to reduce work per-sample, but
need to preserve correctness. While operator reordering is a
well-known optimization technique in database systems [38],
its benefits have yet to be explored for input data pipelines.
Input data pipelines rely on a vast range of domain-specific
libraries and UDFs in contrast to standard SQL clauses, precluding systems from understanding permissible reorderings.
However, reordering UDFs based on how they affect the size
of each sample can unlock significant performance.
To demonstrate this, Figure 6 shows the pipeline execution time across seven “semantically equivalent” reorderings
(i.e., orderings which produce permissible samples for model
training). Reordering can have a massive impact on performance; there is a 5.90× difference in execution time between the most and the least optimal ordering. The ideal reordering pushed size-reducing transformations (e.g., crop and
grayscale) towards the beginning and size-increasing transformations (e.g., int8 to float32 conversion) towards the output.
The key in unlocking these benefits is enabling input data systems to understand which reorderings are permissible, without
requiring users to specify the combinatorial set of reorderings.
Prefetching: Pipeline execution across compute resources.
Prefetching is a powerful tool that enables input data systems to overlap computation across various systems (i.e.,
the accelerator, the data loading process, and offloaded backends). Input data systems can and should leverage prefetching
in two ways. First, prefetching the output of the input data
pipeline allows input data systems to overlap input data pro-

Figure 5: Pipeline execution time and cache size requirement
after materializing the output of each operator (see Figure 2),
normalized to no caching and the raw dataset size (‘-’).
harms execution time because each local process performs
more work facilitating RPCs than it would performing the
operators themselves. That being said, remote backends can
offer benefits if used correctly. mp+sr only selectively offloads
the blur operator, which exhibits high arithmetic intensity, improving execution time by a further 20% over mp.
Fusion: Fuse offloaded operators selectively. Operator fusion can further improve the benefits of offloading operators
by eliminating intermediate data transfer bottlenecks. However, blindly applying fusion (e.g., fusing all map operators)
has the same pitfall as offloading. To illustrate this, Figure 4
shows the execution time of the CV pipeline when fusing and
offloading certain operators to the remote 32-core VM.
In accordance with Figure 3, offloading high arithmetic intensity operators (blur - (R)) improves performance. However,
fusing neighboring operators can either further improve or degrade performance. Fusing grayscale (G) harms performance
as converting images to grayscale locally prior to offloading
reduces sample sizes and I/O costs. Meanwhile, continuing
to fuse compute-intensive operators (namely jitter (J)) significantly reduces execution time (63% of the baseline), achieving the best result. Further fusing crop (C) eliminates these
benefits and incurs slowdowns for the same reason as initially
fusing grayscale. Finding the ideal fusion is non-intuitive and
requires a complex analysis of operator dynamics.
Caching: Decide if and where to cache, without violating operator semantics. Caching can be an effective tool
in trading compute savings for storage costs, but may also
hinder system performance and model accuracy if applied
naively. Figure 5 shows the execution time (over 16 epochs)
and intermediate cache size across ten experiments. In each
4

4

ML Practitioner

Metadata
Store

6
5

cedar
Optimizer

dataset
.register(pipe,...)

2

Execution Interface dataset
.mutate(pipe,...)

3
cedar
Tracer

Raw
Dataset

cedar Source

DataSet Dataflow IR

Optimization
Interface
10
7

dataset
.scale(pipe,...)

9

cedar core

cedar
Driver
cedar
Driver

Execution Backends

cedar
Client

1

to evolve accordingly, each interface exposes primitives that
allow users to quickly define new pipelines, leverage new
compute resources, and integrate new optimizations.
Figure 7 shows an example of cedar in action. (1) An ML
practitioner (MLP) begins by composing a logical input data
pipeline (a Feature) by functionally chaining together operators (Pipes) using the Feature API (§4.1). (2) The MLP also
wraps the raw dataset in a Source (a Pipe abstraction around
a physical dataset) and (3) specifies a set of backends (e.g., the
training node’s CPU, a Kubernetes or Ray cluster, etc.) available for processing. (4) The MLP creates an cedar DataSet,
registering the Feature, Source, and backends. cedar parses
the Feature into a dataflow graph.
(5) cedar’s Optimizer (§5) then statically optimizes
the DataSet using performance statistics for each Pipe
(from past runs or by running a short profiling job), producing an optimized physical plan. (6) The Optimizer updates the Feature’s dataflow graph accordingly using the
Optimization interface (§4.2). (7) Once training begins,
cedar uses the Execution interface (§4.2) to map the optimized dataflow graph’s execution across selected backends.
(8) cedar creates a Client on the training node(s), which uses
a number of Drivers to facilitate processing. cedar initiates
processing and the Client yields processed samples to the
ML training framework (e.g., PyTorch). (9) Meanwhile, the
Client continuously monitors processing, reporting runtime
statistics. (10) The Optimizer uses these statistics to dynamically right-size resources via the Execution interface.

class CustomFeature(Feature):
def compose(self, source): Feature Interface
...

8 GPU

Training Node

Figure 7: cedar block diagram, showing how users can leverage cedar to define, optimize, and execute pipelines.
cessing with the training step. For example, we observed a
35% improvement in end-to-end training throughput for the
CV pipeline using prefetching, given a 100ms training step.
Similarly, prefetching the output of an offloaded operator can
overlap its computation with downstream operators. We observed a 30% improvement in overall input data throughput
using prefetching when offloading the blur operator remotely.
Summary. While numerous optimization techniques can
yield significant performance benefits, input data systems
should be able to concurrently leverage all of these techniques and easily adopt more as they arise. This introduces a
vast search space – applying reordering, caching, offloading
(across 3 backends), and fusion can introduce ∼85B plans for
the CV pipeline. To explore this search space, systems need to
understand the complex systems dynamics and logical semantics of each pipeline and carefully apply each optimization.

4

4.1

Feature API

cedar provides a library of composable, stateless Pipes that
implement common data loading primitives (e.g., batching,
shuffling) and higher-order functions (e.g., map, filter) that
support arbitrary libraries and UDFs. Each Pipe applies a
logical transformation to input samples, yielding transformed
samples to the downstream Pipe(s). Transformations may
be applied one-to-one (e.g., map), many-to-one (e.g., batch),
or one-to-many (e.g., reading lines in a file). Similarly, each
Pipe may ingest (emit) from (to) one or more Pipes (e.g.,
zip/unzip). The Feature API allows users to quickly define an input data pipeline by functionally chaining together
Pipes, composing a dataflow called a Feature.
Figure 8 shows a Feature for the CV pipeline in §3. Each
Feature is logical; it is not bound to a specific dataset,
nor does it specify how the dataflow is executed, allowing
Features to be shared between training jobs. Instead, ML engineers define a Source, which wraps a physical dataset with
a Pipe, providing an iterator over raw samples. Users supply
one or more Sources, the Feature, and backend specifications to the cedar DataSet. The DataSet presents a simple
iterable interface. Training nodes can simply iterate over the
DataSet, which yields fully processed mini-batches within
host memory to be consumed by ML frameworks. Execution

cedar Programming Model and Framework

To address this need, we designed cedar, a framework that
allows ML practitioners to build, optimize, and execute diverse ML input data pipelines. cedar automatically applies
a complex set of optimization techniques to each pipeline,
optimizing its execution over an extensible set of processing
backends with the goal of meeting training jobs’ input data
throughput demands with minimal resource requirements. To
do so, we designed cedar around two key principles.
a) Modularity. Efficiently training ML models requires expertise in multiple domains, including architecting the model,
managing infrastructure, and optimizing performance. cedar
decouples these concerns by presenting three well-defined
interfaces that allow distinct entities to focus on each domain.
b) Composability and Extensibility. Meanwhile, input data
pipelines, execution backends, and performance optimizations will continue to change and improve. To allow cedar
5

Table 1: Optimization (top) & Execution (bottom) interfaces.
API

Description

register(pipe)
fuse(pipes)
update_dfg(graph)
assign(pipe, variant)
set_shards(n)

Registers a new Pipe into the DataSet.
Registers a new Pipe that fuses all input Pipes.
Updates the dataflow graph as specified.
Assign a Variant to the provided Pipe.
Assign a number of Drivers for each Client.

shard(n)
mutate(pipe, variant)
scale(pipe, n)

Shard the Feature into n Drivers.
Mutate the Pipe to the specified Variant.
Sets the parallelism of Pipe to n.

can enforce strict ordering using each sample’s ID.
Figure 8: cedar Feature API.

4.2

is lazy and incremental (i.e., in a streaming fashion), allowing
cedar to scale to large Sources that do not fit in memory.
Relaxed operator dependencies. While the original
Feature dataflow specifies a potential ordering, cedar allows
users to relax ordering constraints by expressing a dependency graph on top of the original dataflow specification. As
shown in Figure 8, if a Pipe v depends on u, users can label u
with an explicit tagu (Line 5) and declare the dependency via
v.depends_on([tagu ]) (Line 6). Users can also fix the position
of a pipe u by calling u. f ix(), which makes u depend on all
upstream Pipes and makes all downstream Pipes depend on
u. As we show in §5, this allows the Optimizer to enumerate
semantically equivalent reordered plans.
Random operators. cedar also allows ML engineers to designate which Pipes represent random augmentations of data
(Lines 5-7), allowing the Optimizer to preserve randomness
while introducing optimization techniques such as caching.
Guarantees and Fault Tolerance. cedar guarantees that the
above constraints are met. cedar also provides fault-tolerance
and precise checkpointing. Each Source wraps emitted samples in a DataSample, each with a UUID. Upon reception of
a batch, the Client records all IDs (aggregations track all aggregated IDs, and filters forward empty DataSamples). To reduce the volume of tracked IDs, Sources map DataSamples
into fixed-size splits. When all samples within a split are received by the Client, the Client seals the split and tracks
the completed split as opposed to all of its samples.
Pipes are stateless. If a fault occurs during execution,
the Client simply informs the Source to re-emit a specific
sample ID, recomputing the result. If duplicate IDs are received, Clients will not return the duplicate sample to ensure exactly-once semantics. Clients expose a checkpointing API, which returns all sealed splits and returned IDs of
open splits. If the training job or Client fails, cedar can resume each Source from the checkpoint. While Sources emit
samples in deterministic order, samples may be received outof-order if backends shuffle samples. While this is acceptable
for most training jobs, which require random shuffling, cedar

Optimization and Execution Interfaces

Once cedar parses the Feature and Sources into the intermediate dataflow graph, it exposes Optimization and
Execution interfaces which allow the cedar Optimizer to
automatically optimize and orchestrate the DataSet’s execution across backends. These dedicated interfaces allow cedar
to easily adopt future optimization techniques and backends.
Optimization interface. The Optimizer (§5) automatically
uses multiple techniques to produce an optimized execution
plan. The Optimization interface provides methods (Table 1) that allow the Optimizer to administer the plan onto
the DataSet’s dataflow. To introduce caching or prefetching,
the Optimizer creates a cache or prefetch Pipe from cedar’s
native library, registers the Pipe via register, and updates
the dataflow graph via update_dfg to insert the Pipe at the
appropriate location. Fusion can be implemented by creating
a new fused Pipe via fuse and updating the dataflow graph
accordingly. Reordering can be implemented by simply updating the dataflow graph. Finally, assign assigns the execution
backend for each Pipe and set_shards sets the number of
Drivers for each Client, as we discuss next.
Execution interface. When the training job starts, cedar
uses the Execution interface to physically map the DataSet
across backends and initiate processing. These methods allow
cedar to leverage parallelism along two dimensions.
First, a Driver is a Python process that coordinates the
DataSet execution (e.g., executing non-offloaded Pipes,
calling backend RPCs, etc.). By default, cedar uses a single Driver for each Client. shard replicates the entire
Feature, creates a Driver for each replica, and shards the
samples generated by each Source across all Drivers. By
creating multiple Drivers, cedar enables two benefits. a)
Users can create multiple Clients, sharding Drivers between each Client to support multi-accelerator training.
b) The Optimizer can create multiple Drivers (using
set_shards) to improve the throughput of a single Client,
parallelizing native Python operators constrained by the GIL.
Secondly, cedar can further offload individual Pipes within
each Driver to an extensible set of backends. Each Pipe is
6

associated with one or more Variants. A Variant represents the physical implementation (e.g., execute the operator
on a distributed cluster using Ray) of the logical operation
specified by the Pipe. cedar calls mutate to transform the
Pipe to the Variant specified by the Optimizer, initiating
processing on the respective backend. Some Variants allow a configurable amount of parallelism (e.g., process pool
size); scale sets the amount of parallelism appropriately.
mutate and scale can also be called dynamically, allowing
the Optimizer to right-size resources during runtime (§ 5.2).

Algorithm 1 The Optimizer’s static optimization passes.

5

11: end for
12: for o f f load_pipe, o f f load_variant in o f f loads do
13:
dataset.assign(o f f load_pipe, o f f load_variant)
14: end for
15: pre f etches, plan ← insert_pre f etch(plan)
16: for pre f etch_pipe in pre f etches do

1: plan ← parsed baseline dataflow graph
2: dataset ← handle to Optimization API
3: plan ← reorder(plan)
4: cache_pipe, plan ← insert_cache(plan)
5: if cache_pipe! = None then
6:
dataset.register(cache_pipe)
7: end if
8: f usions, o f f loads, plan ← f use_and_o f f load(plan)
9: for f used_set in f usions do
10:
dataset. f use( f used_set)

Optimizer

The Optimizer leverages cedar’s core interfaces to statically
optimize and dynamically tune the execution of the DataSet.
Specifically, the Optimizer performs a static pass that concurrently applies the optimizations in Section 3. During runtime,
the Optimizer dynamically adjusts the scale and Variant
of each Pipe to meet throughput demands at minimal cost.
Tracing and profiling. The Optimizer relies on a collection of performance statistics within the Metadata Store
(Figure 7) to calculate cost models across plans. These statistics are automatically collected by cedar, which traces the
execution of each Pipe. Each Source periodically marks
DataSamples to be traced. Each Pipe transparently tags
traced DataSamples with statistics, including the sample’s
execution latency and size (bytes), and the Pipe’s current
Variant and prefetch buffer length (if applicable). Upon
reception, the Client updates the Metadata Store with
traced results and a measure of the current overall throughput.
The Optimizer requires a core set of statistics. These
include the throughput t putbase of the baseline plan, Gbase ,
which executes the un-optimized DataSet locally (i.e., within
a single Driver Python process). For each Pipe p, the
Optimizer also requires the average latency to process a
sample, latbase (p), and its average input and output sample
sizes, sizein (p) and sizeout (p), respectively. Finally, for each
Variant v available for p, the Optimizer requires the average DataSet throughput achieved by offloading only p to v,
t putv (p). If statistics are insufficient (e.g., from a previous
job), the Optimizer runs a short profiling job.

5.1

17:
dataset.register(pre f etch_pipe)
18: end for
19: dataset.set_shards(calculate_shards(plan))
20: dataset.update_d f g(plan)

(§5.2) right-size resources to meet measured demand. Users
can easily customize the cost model to account for additional
parameters (e.g., storage cost) if necessary.
The Optimizer provides a comprehensive cost model to
evaluate the cost of a plan G′ by assigning a cost(p) to each
Pipe p ∈ G′ . A higher cost represents more work and thus
lower performance. The overall cost for G′ is Σ p∈G′ cost(p),
which the Optimizer seeks to minimize. Each optimization
pass extends the cost model, successively building on a baseline cost model derived from the profiled baseline plan, Gbase .
The baseline calculates costbase (p) = f (p)/t putbase , where
f (p) = latbase (p)/(Σi∈Gbase latbase (i)).
Reordering. The Optimizer begins by finding the best
dataflow ordering (Line 3). reorder enumerates all possible
reorderings of the initial plan Gbase . It does so by removing the
output Pipe, o, of Gbase and recursively calculating the set of
all possible reorderings of the shrunk graph. For each shrunk
reordering Gs with output Pipe os , adding o as a successor
to os produces a viable reordering. Furthermore, if o and os
may be reordered according to user constraints, swapping o
and os also produces a viable reordering (i.e., such that o precedes os ). reorder only reorders Pipes within their linear
subgraph (i.e., fix()-ing Pipes with > 1 input or output).
reorder returns the plan with the lowest cost. To calculate the cost of a reordered plan R, the cost model calculates a size scaling factor S(p) = sizeout,base (p)/sizein,base (p),
representing how much p changes the size of each sample. The cost model then computes the new input size of
p ∈ R as sizein,R (p) = sizeraw ∗ Πi∈U S(i), where sizeraw is
the average raw sample size and U ⊂ R is the ordered sequence of all ancestors of p. The cost model calculates

Static Optimization

Algorithm 1 shows how the Optimizer iteratively applies
a set of cost- and rule-based optimization passes to the
dataflow graph, similar to traditional database query optimizers [26, 27, 31, 38]. Within each pass, the Optimizer enumerates permissible plans under user constraints and evaluates
the best plan using a common cost model or a set of rules.
This design allows the Optimizer to be easily extended with
further optimizations. By default, the static passes use a cost
model that aims to maximize input data throughput in order
to avoid data stalls, while its dynamic auto-tuning capabilities
7

the reordered cost costR (p) = (sizein,R (p)/sizein,base (p)) ∗
costbase (p). Since sizein and sizeout trace the average sample size, reorder accounts for operators that change both
selectivity (e.g., filter) and sample size (e.g., crop).
Caching. Next, the Optimizer evaluates if and where to
best cache (i.e., materialize) intermediate data (Line 4).
insert_cache takes the reordered plan and enumerates all
permissible caching locations within the dataflow (prior
to random-labelled operators). insert_cache returns the
lowest-cost plan. To calculate the cost of a plan with a cache
Pipe pcache , the cost model simply sets the cost of any exclusive ancestor p of pcache (i.e., all paths from p to the output
contain pcache ) to zero. To account for I/O costs to read cached
data, cost(pcache ) = d ∗ sizein,R (pcache ), where d is a constant
scaling factor derived from the node’s disk I/O throughput,
and sizein,R (pcache ) is derived using S(p) as in reordering.
Fusion and offloading. The Optimizer considers offloading and fusion concurrently (Line 8). fuse_and_offload
enumerates all possible offloading plans by generating a
set Pi = {(pi , v)|v ∈ V and pi supports v} for each Pipe pi ,
which contains the set of supported Variants for pi within
the user-enabled backends V . The Optimizer first enumerates plans by taking the Cartesian product between all Pi s.
Then, the Optimizer performs all possible fusions for each
plan (e.g., fusing adjacent Pipes assigned the same Variant
if each Pipe supports fusion).
To compare costs between fused-and-offloaded plans, the
cost model first calculates a cost for each pipe p if it is offloaded to a non-base Variant v (i.e., outside the Driver).
It assigns a lower cost costv (p) to Variants that achieve a
higher throughput. The cost model uses the overall speedup
sv (p) = t putv (p)/t putbase of v. It then uses Amdahl’s Law
to solve for the speedup of p on v and linearly scales the
reordered costR (p) accordingly. We constrain costv (p) ≥ 0.
costv (p) = costR (p) ∗

sv (p)−1 − (1 − f (p))
f (p)

Prefetching and Sharding. Finally, the Optimizer applies a set of rules to prefetch and shard the DataSet.
insert_prefetch (Line 15) inserts a prefetching Pipe after
each offloaded (non-base) Variant, as well as at the end of
the dataflow, to allow pipelined execution throughout the input
data pipeline. calculate_shards calculates the ideal number of shards (i.e., Drivers) to use for each Client. To do
so, the Optimizer estimates the throughput of each Driver
using the cost model. If the throughput is over a threshold, the
Optimizer runs a single Driver within each Client process
to avoid introducing an inter-process communication bottleneck. Otherwise, the Optimizer further replicates Drivers
to maximize local parallelism (i.e., creating a Driver for each
host CPU core). The Optimizer updates the dataflow graph
accordingly, which is now ready for execution.
Summary. The Optimizer applies complex optimizations
in discrete passes (like query optimizers), allowing cedar to
search the optimization space in an efficient and extensible
manner. Within each pass, the Optimizer exhaustively enumerates all plans and finds the optimal plan according to the
cost model. The cost model applies some heuristics to support
general UDFs; for example the reordered cost assumes that
processing latency and sample size scale linearly. Users may
customize the cost model for each Pipe if needed (e.g., to use
quadratic scaling). We applied logical passes (e.g., reordering)
prior to physical optimizations (e.g., offloading). However,
later passes are able to review decisions made by prior passes
to avoid sub-optimal paths, as we showed with caching.

5.2

Dynamic Auto-tuning

The Optimizer’s static pass is designed to select the most
performant plan in order to guarantee that accelerators can be
maximally utilized. Once training begins, the Optimizer also
continuously monitors the execution of the DataSet during
runtime, auto-tuning the Variant and parallelism of each
Pipe in order to minimize the resources required to meet
the training job’s throughput demands. The Optimizer does
so by continuously monitoring performance, identifying the
bottleneck Pipe, and tuning the Pipe using hill climbing.
cedar continuously traces and reports runtime performance
(§4). To identify the bottleneck, the Optimizer examines the
prefetch buffer at the DataSet output. If the buffer length is
over a configurable threshold, the input data pipeline is not the
bottleneck. In this case, the Optimizer selects a random Pipe
p with a non-base Variant and scales down its parallelism
by a unit (e.g., one process). Importantly, if the current parallelism of p cannot be further decreased, cedar will mutate p
into the base Variant to avoid over-provisioning resources.
However, if the output buffer is below a threshold, a bottleneck exists. The Optimizer will attempt to scale-up the
parallelism for the bottleneck Pipe pb . It identifies pb by examining the set of all Pipes which were statically assigned a
non-base Variant, P∗ , which represents Pipes that benefit

(1)

Fusion reduces I/O costs between Pipes. To calculate the cost of a fused Pipe p, which fuses
pipes q1 , ..., qn , the cost model calculates the reduction of I/O relative to the un-fused baseline as
io(p) = (sizein,R (q1 ) + sizeout,R (qn ))/(sizein,R (q1 ) +
2 ∗ sizein,R (q2 ) + ... + 2 ∗ sizein,R (qn ) + sizeout,R (qn )).
The cost model then discounts the aggregate costs
of all original Pipes by the relative I/O savings,
cost f used,v (p) = io(p) ∗ (costv (q1 ) + ... + costv (qn )).
Because offloading/fusion affects performance relative
to caching (e.g., offloading may re-compute samples faster
than reading from the cache), fuse_and_offload enumerates plans using the reordered graph with and without the
previously-inserted cache Pipe, returning the best plan according to the cost model. This allows the Optimizer to
revert prior passes if a more optimal solution is found.
8

Table 2: Description of pipelines used to evaluate cedar.
Pipeline

Description

CV-torch

(torchvision) Decode → Float → RandCrop → RandFlip
→ Jitter → Grayscale → Blur → Normalize

CV-tf

(tf.image) Decode → Float → RandCrop → RandFlip
→ Jitter → Grayscale → Blur → Normalize

NLP-torch
NLP-hf-tf
NLP-tf

(torchtext) Read → torch Tokenizer → Truncate → Emb.
(tf.text) Read → HF Tokenizer → Truncate → Emb.
(tf.text) Read → TF Tokenizer → Truncate → Emb.

ASR

(librosa) Decode → Resample → Spectrogram →
Stretch → Time Mask → Freq. Mask → Mel Scale

Figure 9: Execution time (lower is better) for baselines and cedar-local across six pipelines, normalized to
cedar-local for each. Red X’s denote unsupported pipelines
due to a framework mismatch (e.g., CV-tf on torch).

from offloading. First, the Optimizer examines the prefetch
buffer of all p ∈ P∗ that are currently offloaded (i.e., nonbase), selecting the p with the smallest buffer below a threshold. If no such Pipes exist (e.g., if all p ∈ P∗ are mutated to
the base Variant), the Optimizer examines all p ∈ P∗ with
base Variants and selects the p with the largest latbase (p)
– the largest speedup opportunity. Given pb , the Optimizer
will increase its parallelism by a unit, potentially mutating
pb back into a non-base Variant. If the backend’s resources
are exhausted, the Optimizer will scale down another Pipe,
selected randomly, with the same Variant as pb .
The Optimizer continuously performs tuning with a fixed
interval (e.g., 1 minute), ensuring that the minimal set of resources are utilized to meet input data throughput demands.
Because the Optimizer uses cedar’s Execution interface,
cedar’s auto-tuning policy can be easily extended to account
for various deployment scenarios (e.g., to dynamically allocate resources for multiple jobs across a cluster).

6

tions using TF text [70], but used a Hugging Face BPE tokenizer [20]. Finally, NLP-tf used all TF native operators,
including a SentencePiece tokenizer from TF text. We used
the WikiText-103 dataset [46] for all NLP pipelines. ASR
implements SpecAugment [58], used for speech recognition
tasks. In addition to requiring compute-intensive operations,
ASR represents a common case where pipelines use thirdparty domain-specific libraries. ASR was implemented using
librosa [44], and it used the Common Voice dataset [4].

6.1 Optimizing Performance Across Resources
We begin by evaluating cedar’s processing performance
across both local and remote resources compared to stateof-the-art input data systems. Because required input data
throughput varies across model architectures and training infrastructures for a given pipeline, we report the unbounded
performance – assuming an instant training step (we study
auto-tuning in §6.3). Achieving a higher unbounded performance allows input data systems to meet higher throughput
demands given fixed resources, improving end-to-end training
time if data stalls exist. This is also equivalent to requiring
fewer resources to meet a fixed input data throughput demand,
improving overall resource efficiency.
Can cedar optimize the performance of local input data
processing? Since training nodes often offer a large amount
of CPU host resources [55], we first compared cedar against
three state-of-the-art input data frameworks, all performing
processing on a single VM (n2-standard-8 on Google Cloud).
torch used the PyTorch DataLoader [64], configured with
8 workers (i.e., the number of CPU cores). tf used the
TensorFlow tf.data [52] framework, and we allowed tf.data
to auto-tune [39] its parallelism. ray-local used the Ray
Data framework which executed input data processing using Arrow [21] on top of Ray [67]. We compared these
systems against cedar-local using a single Client; we
allowed cedar-local to launch multiple Drivers and disabled caching. We provided cedar-local with a Python mul-

Evaluation

We designed cedar to support the numerous libraries and
frameworks currently used across ML deployments. Since ML
practitioners predominantly rely on Python, we built cedar
from the ground up in ∼12K lines of Python code. cedar
supports all popular ML frameworks (e.g., PyTorch, JAX,
and TensorFlow) and can execute operations from arbitrary
preprocessing libraries, given a Python API.
Workloads. We evaluated cedar on a diverse set of six ML
input data pipelines across computer vision, natural language,
and speech domains as shown in Table 2. CV-torch is the
SimCLR [13, 14] pipeline in Figure 2, implemented using
torchvision, while CV-tf reimplemented the same operations
using native TensorFlow (TF) image [72] operators. SimCLR
is a common vision task requiring compute-intensive operations and thus performant input data systems [28]. We used
ImageNet [19] as the dataset for both CV pipelines. The NLP
pipelines are based on GPT-2 [66] and represent a typical language pipeline (e.g., for fine-tuning LLMs [11]). NLP-torch
was implemented using torchtext with a Byte Pair Encoding
(BPE) tokenizer. NLP-hf-tf implemented the same opera9

tiprocessing pool backend. To demonstrate the benefits of
cedar’s composability, we also provided backends that allow
cedar to offload portions of the pipeline to tf.data or Ray Data
for processing. Datasets were stored on the VM’s filesystem.
Figure 9 shows the execution time to process the dataset,
normalized to the cedar-local result for each pipeline
(lower is better). torch and tf pipelines were restricted to
their respective frameworks. For the CV pipelines, cedar was
able to improve the throughput compared to all other systems. This was because cedar reordered size-reducing operators (crop, grayscale) towards the beginning of the pipeline,
and size-increasing operators (float) towards the end of the
pipeline. This significantly reduced work for each sample
while obeying dependency constraints (crop → flip, float
→ normalize). cedar also leveraged parallelism by using
8 Drivers. cedar-local obtained a similar performance to
tf due to tf’s graph optimizer and performant C++ backend.
For NLP-torch and NLP-hf-tf, cedar fused Read, Tokenize, and Truncate into one Pipe, which was offloaded to
a multiprocess backend. Meanwhile, cedar spawned only
one Driver within the main Client process. This avoided
a significant serialization bottleneck in sending large embeddings between processes, which slowed down torch
and ray-local on NLP-torch. For NLP-hf-tf, cedar outperformed tf because tf could not parallelize the non-native
Tokenizer operator. However, cedar achieved a slightly lower
performance (0.85× ray-local), as its Arrow core allowed
it to zero-copy intermediate NumPy arrays. cedar’s extensibility allows it to adopt Arrow as a future Variant.
For NLP-tf, tf was able to automatically parse and
compile the pipeline into a heavily-optimized TF graph.
In fact, cedar recognized the performance benefit and offloaded execution to tf.data, without requiring users to change
the Feature, allowing cedar to near tf (0.91× due to
DataSample overheads) and out-perform ray-local. For
ASR, cedar identified that it performed many array copies between operators. cedar pushed Stretch later (to decrease sample sizes) and offloaded execution to the Ray Data backend,
obtaining its zero-copy benefits and matching ray-local’s
performance. Meanwhile, torch and tf suffered from copy
overheads. tf also could not trace the pipeline as a graph and
executed all operators within its GIL-bound Python process.
Figure 9 shows how cedar’s extensive set of optimizations can optimize the underlying dataflow, maximize the
amount of available parallelism while reducing system overheads (e.g., serialization), identify the best execution backend,
and support a diverse set of pipelines, all without requiring
user configuration. These techniques allow cedar to achieve
state-of-the-art performance using trainers’ local CPUs, outperforming tf.data, Ray Data, and PyTorch Dataloaders by
2.49×, 2.18×, and 2.74× on average, respectively.
Can cedar optimize distributed input data processing?
We next compared cedar to two distributed baselines, providing a remote 32-core VM (n2-standard-32) in addition to

Figure 10: Execution time (lower is better) for baselines
and cedar-remote across six pipelines, normalized to
cedar-remote for each. tf-service is unable to support
pipelines with non-native operators (denoted by Red X’s).
the 8-core local VM. tf-service used tf.data service [7] to
disaggregate processing to the remote VM; as with tf, we enabled auto-tuning. ray-remote created a Ray Cluster across
both VMs and used Ray Data. We provided cedar-remote
with a distributed backend that performs remote processing using Ray [49] to execute RPCs. To standardize data
reading across all baselines, we allowed each system to read
raw data from either the local or remote VM’s filesystem.
tf-service read from the remote VM since all operators
were executed remotely. ray-remote spawned tasks on both
VMs; each task read from its respective filesystem. Meanwhile, cedar-remote selectively offloaded reading, depending on the ideal plan for each pipeline. As with cedar-local,
we disabled caching and compared unbounded throughput.
Figure 10 shows the normalized execution time (lower is
better) for each pipeline. Because tf.data service distributes
TF graphs to remote workers, it is unable to support nonnative pipelines that cannot be traced. For the CV pipelines,
cedar generated a similar reordering to cedar-local and
used 8 Drivers. It also offloaded the entirety of the pipeline
to the distributed backend, except for the float conversion and
normalization. Doing so required a quarter of the amount of
I/O (int8 versus float32) compared to offloading the entire
pipeline. In contrast, ray-remote incurred additional I/O
(not selectively offloading operators) and compute costs (not
reordering) on both pipelines. For CV-tf, TensorFlow’s graph
optimizations allowed tf-service to match cedar.
For NLP-torch and NLP-hf-tf, cedar offloaded the Read,
Tokenize, and Truncate operators to a fused distributed Pipe.
As with cedar-local, cedar performed the embedding operator in the single-Driver Client, avoiding serialization
overheads. Interestingly, cedar correctly identified that not
using the remote VM was the optimal plan for NLP-tf, instead offloading processing to a local tf.data backend as with
cedar-local. Meanwhile, tf-service did distribute processing and was bottlenecked by reading large embeddings
over the network. cedar outperformed ray-remote because
ray-remote not only inefficiently read large embeddings
10

Table 3: Auto-cached throughput (normalized to
cedar-remote, higher is better) and throughput of the
next best cache location across the three torch pipelines.
Pipeline

Norm. Throughput with Caching

Next Best Throughput

CV-torch
NLP-torch
ASR

1.74
1.42
1.00 (Do Not Cache)

1.23
1.37
0.82

Table 4: Steady-state throughput (samples/s) for a static number of offloaded processes for the CV-torch pipeline.
Figure 11: Ablation study showing execution time across
pipelines as optimization techniques are successively enabled,
normalized to the unoptimized pipeline. P = local parallelism,
R = reordering, O = offloading, F = fusion.

Samples/s

1 Proc.

2 Procs.

4 Procs.

8 Procs.

70.77

71.43

128.61

245.11

438.91

provements with each optimization showcase cedar’s ability
to incrementally reason about additional techniques.
Can cedar automatically cache to maximize throughput?
We also evaluated if cedar can reason about if and where
caching would be beneficial. To do so, we allowed cedar to
automatically place a Cache Pipe, which materialized all intermediate samples to disk in the first epoch. We then ran
multiple epochs of the torch pipelines across the three domains (CV-torch, NLP-torch, and ASR). Table 3 shows the
throughput (higher is better) of the cached plan after the first
epoch, normalized to the throughput achieved by the plan
generated by cedar-remote. We also report the next-best
throughput achieved by enumerating all other cache locations.
cedar was able to find the optimal location to apply caching.
For CV-torch, cedar cached the result after decode and
grayscale, but prior to random cropping, satisfying randomness requirements. This improved throughput by reducing
disk I/O (grayscale) and compute (decode). For NLP-torch,
cedar cached the tokenized and truncated sample prior to
embedding, avoiding overheads of reading large embeddings.
Interestingly, cedar did not cache ASR, determining that recomputation was ideal because transforms significantly increased data volumes. Table 3 confirms this; caching the output of the entire pipeline reduced throughput by 18%. This
shows that cedar can reason about complex interactions between optimizations, e.g., caching and offloading.

over RPCs, but it also held embeddings within its object store,
quickly saturating it and causing frequent spills to disk. Finally, cedar automatically determined that reordering and
offloading ASR to a Ray Data backend (distributed across
both VMs) was ideal for the same zero-copy benefits as
cedar-local, allowing it to match ray-remote.
cedar successfully optimized distributed processing, selectively offloading operators across both local and remote
resources. On average, cedar outperformed ray-remote and
tf-service by 2.19× and 1.87×, respectively.

6.2

Local

Understanding Optimizations

How do each of cedar’s optimizations contribute to performance? To understand cedar’s ability to compose optimizations, we performed an ablation study by successively
enabling local parallelism (i.e., multiple Drivers), reordering, offloading (with the distributed backend enabled), and
fusion. All experiments used prefetching. Figure 11 shows
the execution time of each experiment, normalized to the
baseline (i.e., executing the unoptimized DataSet within a
single Driver). No single optimization is a panacea. Instead,
a diverse set of optimizations is needed to achieve high performance due to the diverse characteristics across pipelines.
Local parallelism was effective for CV-torch and ASR
since using multiple Drivers bypassed their GIL bottleneck.
Meanwhile, CV-tf largely executed in TensorFlow’s C++
backend. cedar did not parallelize the NLP pipelines to avoid
serialization overheads. Reordering was effective at improving both CV pipelines since they used volume-changing operators. Meanwhile, ASR had more limited reordering opportunities, and the NLP pipelines were not able to be reordered. All
pipelines ultimately took advantage of offloading across backends. However, Figure 11 shows how the overheads incurred
by data movement across offloaded pipes limited its effectiveness for some pipelines. By eliminating these overheads with
fusion, cedar was ultimately able to improve throughput by
2.54 − 43.82× compared to the baseline. The successive im-

6.3

Auto-tuning Parallelism

Can cedar dynamically right-size resources? By default,
cedar assigns a default level of parallelism for each offloaded
Pipe (e.g., dividing all available resources among Pipes
offloaded to each respective Variant), intending to maximize throughput and eliminate data stalls. However, this may
overprovision resources if the training node requires a lower
throughput. To demonstrate this, Table 4 shows the throughput
achieved by a single Driver given a fixed number of remote
distributed processes used to offload the fused operator
for CV-torch in the cedar-remote experiment.
cedar can right-size the amount of resources required to
11

Table 5: Number of processes tuned to by the Optimizer to
reach a target training throughput (samples/s) for CV-torch.
Training Throughput

Auto-Tuned # Processes

39.83
194.40
379.13
559.11 [Unbounded]

0 [Used base Variant]
4
8
12

itly label permissible caching locations within the pipeline,
automatically selecting the best one; however this API is incompatible with reordering. Plumber [39] can detect random
operators by tracing tf.data graphs and automatically allocates
resources for parallelism and caching, but is limited to a single node. DPP [82], GoldMiner [81], tf.data [52], and tf.data
service [7] implement auto-scaling policies.
CoorDL [48], OneAccess [36], Quiver [40], and SiloD [80]
implement distributed caches for input data shared across
training jobs. PRESTO [34] is a profiling tool that determines
the ideal location to cache. Revamper [43] caches partiallyaugmented data, reasoning about model accuracy. Similarly,
SHADE [37] and iCACHE [15] up-sample and cache useful
training samples. Tectonic-Shift [83] is a cache deployed at
Meta for ML datasets. Various systems leverage specialized
file types to improve I/O efficiency for ML datasets [56,73,74,
76]. NVIDIA DALI [54] offloads certain input data operators
onto GPUs. GoldMiner [81] can balance operators between
the input data pipeline and training step, offloading operators
to the CPU or GPU, respectively. FastFlow [75] splits input
data processing between local and remote workers at a coarse
granularity, selecting from three candidate plans that perform
reading, transformations, and batching locally or remotely.
cedar’s extensible Optimizer allows users to systematically improve performance as new techniques arise.
Traditional Processing Frameworks. Naiad [51], Spark [78],
DryadLINQ [77], and many other processing frameworks [2,
3, 10, 18, 33, 45] allow users to chain together higher-order
transformations and model computation as data flows. cedar’s
Execution interface shares a similar methodology as Apache
Beam [2, 23], which allows logical dataflow graphs to be
executed on a number of execution backends via a Runner
interface. While Beam lowers the entire graph to a specific
backend, cedar can concurrently mutate individual Pipes
to different Variants, as well as dynamically mutate the
Variant of a Pipe to right-size resources.
cedar applies similar rule- and cost-based optimization
passes to traditional query optimizers [6, 26, 27], while also
considering unique properties to ML input data pipelines such
as relaxed order dependencies and randomness. Lara [41] is
a domain-specific language that optimizes matrix-heavy preprocessing for traditional “shallow” ML models (e.g., regressions). Hueske et al. [31] explore preserving UDF semantics
under reordering by statically analyzing PACT [9] programs.

meet the required throughput for each training job by dynamically tuning both the scale and Variant of each Pipe (§5).
To evaluate its effectiveness, we simulated a training job by
requesting samples at fixed rates, as shown in the left column of Table 5. We then configured cedar to begin with 1
distributed process and allowed cedar to tune its scale;
the right column reports the steady-state number of processes.
cedar can dynamically mutate a Pipe’s Variant from an
offloaded backend into the Driver process (and vice versa).
This allowed cedar to completely free the remote VM in the
case of low training throughput as opposed to holding it with
one process. Furthermore, even within a Variant, cedar is
able to minimize the amount of held resources needed to meet
a requested throughput. These results show the efficiency
enabled by cedar’s dynamic traces and Execution interface.

7

Related Work

ML Input Data Systems. PyTorch DataLoaders [64] and
tf.data [52] are the de facto input data frameworks for their
respective ML framework. DataLoaders allow users to specify
a number of local workers, while tf.data can statically fuse and
vectorize TensorFlow-native pipelines. tf.data can also autotune the amount of CPU and RAM allocated to each operator.
tf.data service offloads tf.data processing to disaggregated
workers, but it cannot support non-TensorFlow UDFs [71].
TorchData [63] is a beta PyTorch-native data loading library and provides similar primitives to Pipes. TorchData’s
DataLoader2 [61] is an incomplete prototype that includes
an API for distributed processing. Unfortunately, active development on TorchData has stopped [62]. DPP [82] and
GoldMiner [81] are proprietary distributed services deployed
at Meta and Alibaba, respectively. Ray Data [67] is an input
data library built on top of Ray [49]. Ray Data distributes processing using Ray’s Task and Actor primitives, and optimizes
for task overheads by fusing operators via fixed rules.
cedar supports arbitrary libraries and ML frameworks, including PyTorch and TensorFlow. It provides a similar programming interface to current frameworks, while additionally
permitting users to express a simple set of constraints that
unlock a rich set of reordering and caching optimizations.
cedar further provides an extensible Optimizer, which transparently applies powerful optimizations across backends.
ML Input Data Optimizations. Cachew [28], built on top of
tf.data, automatically scales the number of distributed workers
for a given training job. Cachew also allows users to explic-

8

Conclusion

We presented cedar, a composable and optimized framework
to define, optimize, and execute ML input data pipelines.
cedar’s Feature API allows users to define input data
pipelines using modular Pipes and to express lightweight
hints that allow cedar to reason about operator semantics.
cedar models the pipeline as a dataflow graph, exposing a
set of interfaces that allow cedar’s Optimizer to optimize
12

and execute the pipeline across a configurable set of backends. We demonstrated how this allows cedar to outperform
tf.data, tf.data service, Ray Data, and PyTorch DataLoaders by
2.49×, 1.87×, 2.18×, and 2.74× on average across diverse
pipelines.

[6] Michael Armbrust, Reynold S. Xin, Cheng Lian, Yin
Huai, Davies Liu, Joseph K. Bradley, Xiangrui Meng,
Tomer Kaftan, Michael J. Franklin, Ali Ghodsi, and
Matei Zaharia. Spark sql: Relational data processing
in spark. In Proceedings of the 2015 ACM SIGMOD
International Conference on Management of Data, SIGMOD ’15, page 1383–1394, New York, NY, USA, 2015.
Association for Computing Machinery.

References
[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene
Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado,
Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit
Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin
Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow:
Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.

[7] Andrew Audibert, Yang Chen, Dan Graur, Ana
Klimovic, Jiří Šimša, and Chandramohan A. Thekkath.
Tf.data service: A case for disaggregating ml input data
processing. In Proceedings of the 2023 ACM Symposium on Cloud Computing, SoCC ’23, page 358–375,
New York, NY, USA, 2023. Association for Computing
Machinery.
[8] AWS. Aws trainium. https://aws.amazon.com/
machine-learning/trainium/, 2022.
[9] Dominic Battré, Stephan Ewen, Fabian Hueske, Odej
Kao, Volker Markl, and Daniel Warneke. Nephele/pacts:
A programming model and execution framework for
web-scale analytical processing. In Proceedings of the
1st ACM Symposium on Cloud Computing, SoCC ’10,
page 119–130, New York, NY, USA, 2010. Association
for Computing Machinery.

[2] Tyler Akidau, Robert Bradshaw, Craig Chambers, Slava
Chernyak, Rafael J. Fernández-Moctezuma, Reuven
Lax, Sam McVeety, Daniel Mills, Frances Perry, Eric
Schmidt, and Sam Whittle. The dataflow model: A
practical approach to balancing correctness, latency, and
cost in massive-scale, unbounded, out-of-order data processing. Proceedings of the VLDB Endowment, 8:1792–
1803, 2015.

[10] Vinayak Borkar, Michael Carey, Raman Grover, Nicola
Onose, and Rares Vernica. Hyracks: A flexible and extensible foundation for data-intensive computing. In Proceedings of the 2011 IEEE 27th International Conference on Data Engineering, ICDE ’11, page 1151–1162,
USA, 2011. IEEE Computer Society.

[3] Alexander Alexandrov, Rico Bergmann, Stephan Ewen,
Johann-Christoph Freytag, Fabian Hueske, Arvid Heise,
Odej Kao, Marcus Leich, Ulf Leser, Volker Markl,
Felix Naumann, Mathias Peters, Astrid Rheinländer,
Matthias J. Sax, Sebastian Schelter, Mareike Höger,
Kostas Tzoumas, and Daniel Warneke. The stratosphere
platform for big data analytics. The VLDB Journal,
23(6):939–964, dec 2014.

[11] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen
Krueger, Tom Henighan, Rewon Child, Aditya Ramesh,
Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin,
Scott Gray, Benjamin Chess, Jack Clark, Christopher
Berner, Sam McCandlish, Alec Radford, Ilya Sutskever,
and Dario Amodei. Language models are few-shot learners, 2020.

[4] Rosana Ardila, Megan Branson, Kelly Davis, Michael
Henretty, Michael Kohler, Josh Meyer, Reuben Morais,
Lindsay Saunders, Francis M. Tyers, and Gregor Weber. Common voice: A massively-multilingual speech
corpus, 2020.
[5] Michael Armbrust, Tathagata Das, Liwen Sun, Burak
Yavuz, Shixiong Zhu, Mukul Murthy, Joseph Torres,
Herman van Hovell, Adrian Ionescu, Alicja Łuszczak,
Michał undefinedwitakowski, Michał Szafrański, Xiao
Li, Takuya Ueshin, Mostafa Mokhtar, Peter Boncz, Ali
Ghodsi, Sameer Paranjpye, Pieter Senster, Reynold Xin,
and Matei Zaharia. Delta lake: High-performance acid
table storage over cloud object stores. Proc. VLDB
Endow., 13(12):3411–3424, aug 2020.

[12] P Carbone, S Ewen, S Haridi, A Katsifodimos, V Markl,
and K Tzoumas. Apache flinktm: Stream and batch
processing in a single engine. Bull. IEEE Comput. Soc.
Tech. Comm. Data Eng, 36(4), 2015.
[13] Ting Chen, Simon Kornblith, Mohammad Norouzi, and
Geoffrey Hinton. A simple framework for contrastive
learning of visual representations. arXiv preprint
arXiv:2002.05709, 2020.
13

[14] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised
models are strong semi-supervised learners. arXiv
preprint arXiv:2006.10029, 2020.

[26] G. Graefe and W.J. McKenna. The volcano optimizer
generator: extensibility and efficient search. In Proceedings of IEEE 9th International Conference on Data
Engineering, pages 209–218, 1993.

[15] Weijian Chen, Shuibing He, Yaowen Xu, Xuechen
Zhang, Siling Yang, Shuang Hu, Xian-He Sun, and Gang
Chen. icache: An importance-sampling-informed cache
for accelerating i/o-bound dnn model training. In 2023
IEEE International Symposium on High-Performance
Computer Architecture (HPCA), pages 220–232, 2023.

[27] Goetz Graefe. The cascades framework for query optimization. IEEE Data Eng. Bull., 18(3):19–29, 1995.
[28] Dan Graur, Damien Aymon, Dan Kluser, Tanguy Albrici, Chandramohan A. Thekkath, and Ana Klimovic.
Cachew: Machine learning input data processing as a
service. In 2022 USENIX Annual Technical Conference
(USENIX ATC 22), pages 689–706, Carlsbad, CA, July
2022. USENIX Association.

[16] Ekin Dogus Cubuk, Barret Zoph, Jon Shlens, and Quoc
Le. Randaugment: Practical automated data augmentation with a reduced search space. In H. Larochelle,
M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors,
Advances in Neural Information Processing Systems,
volume 33, pages 18613–18624. Curran Associates, Inc.,
2020.

[29] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. International Conference on Learning Representations (ICLR),
2016.

[17] Benoit Dageville, Thierry Cruanes, Marcin Zukowski,
Vadim Antonov, Artin Avanes, Jon Bock, Jonathan Claybaugh, Daniel Engovatov, Martin Hentschel, Jiansheng
Huang, Allison W. Lee, Ashish Motivala, Abdul Q. Munir, Steven Pelley, Peter Povinec, Greg Rahn, Spyridon
Triantafyllis, and Philipp Unterbrunner. The snowflake
elastic data warehouse. In Proceedings of the 2016
International Conference on Management of Data, SIGMOD ’16, page 215–226, New York, NY, USA, 2016.
Association for Computing Machinery.

[30] Song Han, Jeff Pool, John Tran, and William J. Dally.
Learning both weights and connections for efficient neural networks. In Proceedings of the 28th International
Conference on Neural Information Processing Systems Volume 1, NIPS’15, page 1135–1143, Cambridge, MA,
USA, 2015. MIT Press.
[31] Fabian Hueske, Mathias Peters, Matthias J. Sax, Astrid
Rheinländer, Rico Bergmann, Aljoscha Krettek, and
Kostas Tzoumas. Opening the black boxes in data flow
optimization. Proc. VLDB Endow., 5(11):1256–1267,
jul 2012.

[18] Jeffrey Dean and Sanjay Ghemawat. Mapreduce: Simplified data processing on large clusters. Commun. ACM,
51(1):107–113, jan 2008.

[32] HuggingFace.
Tokenizer summary.
//huggingface.co/docs/transformers/
tokenizer_summary, 2023.

[19] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical
image database. In 2009 IEEE Conference on Computer
Vision and Pattern Recognition, pages 248–255, 2009.

https:

[33] Michael Isard, Mihai Budiu, Yuan Yu, Andrew Birrell,
and Dennis Fetterly. Dryad: Distributed data-parallel
programs from sequential building blocks. In Proceedings of the 2nd ACM SIGOPS/EuroSys European Conference on Computer Systems 2007, EuroSys ’07, page
59–72, New York, NY, USA, 2007. Association for Computing Machinery.

Transformers.
https:
[20] Hugging Face.
//huggingface.co/docs/transformers/index,
2023.
[21] Apache Software Foundation. Apache arrow. https:
//arrow.apache.org, 2023.

[34] Alexander Isenko, Ruben Mayer, Jeffrey Jedele, and
Hans-Arno Jacobsen. Where is my training bottleneck? hidden trade-offs in deep learning preprocessing
pipelines. In Proceedings of the 2022 International Conference on Management of Data, SIGMOD ’22, page
1825–1839, New York, NY, USA, 2022. Association for
Computing Machinery.

[22] Apache Software Foundation. Apache avro. https:
//avro.apache.org, 2023.
[23] Apache Software Foundation. Apache beam. https:
//beam.apache.org, 2023.
[24] Apache Software Foundation. Apache orc. https:
//orc.apache.org, 2023.

[35] Norm Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul
Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, Clifford Young, Xiang

[25] Apache Software Foundation. Apache parquet. https:
//parquet.apache.org/docs/, 2023.
14

Zhou, Zongwei Zhou, and David A Patterson. Tpu v4:
An optically reconfigurable supercomputer for machine
learning with hardware support for embeddings. In Proceedings of the 50th Annual International Symposium
on Computer Architecture, ISCA ’23, New York, NY,
USA, 2023. Association for Computing Machinery.

[44] Brian McFee, Colin Raffel, Dawen Liang, Daniel P Ellis,
Matt McVicar, Eric Battenberg, and Oriol Nieto. librosa:
Audio and music signal analysis in python. In Proceedings of the 14th Python in Science Conference, volume 8,
pages 18–25, 2015.
[45] Erik Meijer, Brian Beckman, and Gavin Bierman. Linq:
Reconciling object, relations and xml in the .net framework. In Proceedings of the 2006 ACM SIGMOD International Conference on Management of Data, SIGMOD
’06, page 706, New York, NY, USA, 2006. Association
for Computing Machinery.

[36] Aarati Kakaraparthy, Abhay Venkatesh, Amar Phanishayee, and Shivaram Venkataraman. The case for
unifying data loading in machine learning clusters. In
11th USENIX Workshop on Hot Topics in Cloud Computing (HotCloud 19), Renton, WA, July 2019. USENIX
Association.

[46] Stephen Merity, Caiming Xiong, James Bradbury, and
Richard Socher. Pointer sentinel mixture models, 2016.

[37] Redwan Ibne Seraj Khan, Ahmad Hossein Yazdani, Yuqi
Fu, Arnab K. Paul, Bo Ji, Xun Jian, Yue Cheng, and
Ali R. Butt. SHADE: Enable fundamental cacheability
for distributed deep learning training. In 21st USENIX
Conference on File and Storage Technologies (FAST
23), pages 135–152, Santa Clara, CA, February 2023.
USENIX Association.

[47] Meta. Introducing the ai research supercluster. https:
//ai.facebook.com/blog/ai-rsc/, 2022.
[48] Jayashree Mohan, Amar Phanishayee, Ashish Raniwala,
and Vijay Chidambaram. Analyzing and mitigating
data stalls in dnn training. Proc. VLDB Endow.,
14(5):771–784, jan 2021.

[38] Jan Kossmann, Thorsten Papenbrock, and Felix Naumann. Data dependencies for query optimization: A
survey. The VLDB Journal, 31(1):1–22, jun 2021.

[49] Philipp Moritz, Robert Nishihara, Stephanie Wang,
Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan,
and Ion Stoica. Ray: A distributed framework for
emerging ai applications. In Proceedings of the 13th
USENIX Conference on Operating Systems Design and
Implementation, OSDI’18, page 561–577, USA, 2018.
USENIX Association.

[39] Michael Kuchnik, Ana Klimovic, Jiri Simsa, Virginia
Smith, and George Amvrosiadis. Plumber: Diagnosing
and removing performance bottlenecks in machine learning data pipelines. Proceedings of Machine Learning
and Systems, 4:33–51, 2022.

[50] Dheevatsa Mudigere, Yuchen Hao, Jianyu Huang, Zhihao Jia, Andrew Tulloch, Srinivas Sridharan, Xing Liu,
Mustafa Ozdal, Jade Nie, Jongsoo Park, Liang Luo,
Jie (Amy) Yang, Leon Gao, Dmytro Ivchenko, Aarti
Basant, Yuxi Hu, Jiyan Yang, Ehsan K. Ardestani, Xiaodong Wang, Rakesh Komuravelli, Ching-Hsiang Chu,
Serhat Yilmaz, Huayu Li, Jiyuan Qian, Zhuobo Feng,
Yinbin Ma, Junjie Yang, Ellie Wen, Hong Li, Lin Yang,
Chonglin Sun, Whitney Zhao, Dimitry Melts, Krishna
Dhulipala, KR Kishore, Tyler Graf, Assaf Eisenman, Kiran Kumar Matam, Adi Gangidi, Guoqiang Jerry Chen,
Manoj Krishnan, Avinash Nayak, Krishnakumar Nair,
Bharath Muthiah, Mahmoud khorashadi, Pallab Bhattacharya, Petr Lapukhov, Maxim Naumov, Ajit Mathews,
Lin Qiao, Mikhail Smelyanskiy, Bill Jia, and Vijay Rao.
Software-hardware co-design for fast and scalable training of deep learning recommendation models. In Proceedings of the 49th Annual International Symposium
on Computer Architecture, ISCA ’22, page 993–1011,
New York, NY, USA, 2022. Association for Computing
Machinery.

[40] Abhishek Vijaya Kumar and Muthian Sivathanu. Quiver:
An informed storage cache for deep learning. In 18th
USENIX Conference on File and Storage Technologies
(FAST 20), pages 283–296, Santa Clara, CA, February
2020. USENIX Association.
[41] Andreas Kunft, Asterios Katsifodimos, Sebastian Schelter, Sebastian Breß, Tilmann Rabl, and Volker Markl.
An intermediate representation for optimizing machine learning pipelines.
Proc. VLDB Endow.,
12(11):1553–1567, jul 2019.
[42] Frederic Lardinois. Google launches a 9 exaflop
cluster of cloud TPU v4 pods into public preview.
https://techcrunch.com/2022/05/11/googlelaunches-a-9-exaflop-cluster-of-cloud-tpuv4-pods-into-public-preview/, 2022.
[43] Gyewon Lee, Irene Lee, Hyeonmin Ha, Kyunggeun Lee,
Hwarim Hyun, Ahnjae Shin, and Byung-Gon Chun. Refurbish your training data: Reusing partially augmented
samples for faster deep neural network training. In 2021
USENIX Annual Technical Conference (USENIX ATC
21), pages 537–550. USENIX Association, July 2021.

[51] Derek G. Murray, Frank McSherry, Rebecca Isaacs,
Michael Isard, Paul Barham, and Martín Abadi. Na15

iad: A timely dataflow system. In Proceedings of the
Twenty-Fourth ACM Symposium on Operating Systems
Principles, SOSP ’13, page 439–455, New York, NY,
USA, 2013. Association for Computing Machinery.

[61] PyTorch.
Dataloader2.
https:
//github.com/pytorch/data/blob/
a5b4720dece60565788ac4c9a85e01719188b28e/
torchdata/dataloader2/README.md, 2023.

[52] Derek G. Murray, Jiří Šimša, Ana Klimovic, and Ihor
Indyk. Tf.data: A machine learning data processing
framework. Proc. VLDB Endow., 14(12):2945–2958,
jul 2021.

[62] PyTorch. Future of torchdata and dataloading. https:
//github.com/pytorch/data/issues/1196, 2023.
[63] PyTorch. Torchdata. https://pytorch.org/data/
beta/index.html, 2023.

numpy.lib.format.
https://
[53] NumPy.
numpy.org/devdocs/reference/generated/
numpy.lib.format.html#, 2023.

[64] PyTorch. torch.utils.data. https://pytorch.org/
docs/stable/data.html, 2023.

[54] NVIDIA. Nvidia dali. https://docs.nvidia.com/
deeplearning/dali/user-guide/docs/
index.html, 2023.

[65] PyTorch.
Torchvision.
https://pytorch.org/
vision/stable/index.html, 2023.
[66] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario
Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

[55] NVIDIA.
Nvidia dgx h100.
https://
www.nvidia.com/en-us/data-center/dgx-h100/,
2023.

[67] Ray. Ray data: Scalable datasets for ml. https://
docs.ray.io/en/latest/data/data.html, 2023.

[56] NVIDIA. Nvtabular. https://github.com/NVIDIAMerlin/NVTabular, 2023.

Nvidia hopper, ampere gpus
[68] Dave Salvator.
sweep benchmarks in ai training.
https:
//blogs.nvidia.com/blog/2022/11/09/mlperfai-training-hpc-hopper/, 2023.

[57] Satadru Pan, Theano Stavrinos, Yunqiao Zhang,
Atul Sikaria, Pavel Zakharov, Abhinav Sharma,
Shiva Shankar P, Mike Shuey, Richard Wareing,
Monika Gangapuram, Guanglei Cao, Christian Preseau,
Pratap Singh, Kestutis Patiejunas, JR Tipton, Ethan
Katz-Bassett, and Wyatt Lloyd. Facebook’s tectonic
filesystem: Efficiency from exascale. In 19th USENIX
Conference on File and Storage Technologies (FAST
21), pages 217–231. USENIX Association, February
2021.

[69] Geet Sethi, Bilge Acun, Niket Agarwal, Christos
Kozyrakis, Caroline Trippel, and Carole-Jean Wu. Recshard: Statistical feature-based memory optimization for
industry-scale neural recommendation. In Proceedings
of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS ’22, page 344–358, New York,
NY, USA, 2022. Association for Computing Machinery.

[58] Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng
Chiu, Barret Zoph, Ekin D. Cubuk, and Quoc V. Le.
SpecAugment: A Simple Data Augmentation Method
for Automatic Speech Recognition. In Proc. Interspeech
2019, pages 2613–2617, 2019.

[70] TensorFlow.
Tensorflow text.
//www.tensorflow.org/text, 2023.

https:

[71] TensorFlow.
tf.data.experimental.service.
https://www.tensorflow.org/api_docs/python/
tf/data/experimental/service#limitations,
2023.

[59] Adam Paszke, Sam Gross, Francisco Massa, Adam
Lerer, James Bradbury, Gregory Chanan, Trevor Killeen,
Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban
Desmaison, Andreas Köpf, Edward Yang, Zach DeVito,
Martin Raison, Alykhan Tejani, Sasank Chilamkurthy,
Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style, High-Performance
Deep Learning Library. Curran Associates Inc., Red
Hook, NY, USA, 2019.

[72] TensorFlow.
tf.image.
https://
www.tensorflow.org/api_docs/python/tf/image,
2023.
[73] TensorFlow. Tfrecord and tf.train.example. https:
//www.tensorflow.org/tutorials/load_data/
tfrecord#tfrecord_files_using_tfdata, 2023.

[60] Pedro Pedreira, Orri Erling, Masha Basmanova, Kevin
Wilfong, Laith Sakka, Krishna Pai, Wei He, and
Biswapesh Chattopadhyay. Velox: Meta’s unified execution engine. Proc. VLDB Endow., 15(12):3372–3384,
aug 2022.

[74] Uber.
Petastorm.
https://
petastorm.readthedocs.io/en/latest/
index.html, 2023.
16

[75] Taegeon Um, Byungsoo Oh, Byeongchan Seo, Minhyeok Kweun, Goeun Kim, and Woo-Yeon Lee. Fastflow: Accelerating deep learning model training with
smart offloading of input data pipeline. Proc. VLDB
Endow., 16(5):1086–1099, jan 2023.

ommendation model training: Industrial product. In Proceedings of the 49th Annual International Symposium
on Computer Architecture, ISCA ’22, page 1042–1057,
New York, NY, USA, 2022. Association for Computing
Machinery.

[76] Suketu Vakharia, Peng Li, Weiran Liu, and Sundaram
Narayanan. Shared foundations: Modernizing meta’s
data lakehouse. In The Conference on Innovative Data
Systems Research, CIDR, 2023.

[83] Mark Zhao, Satadru Pan, Niket Agarwal, Zhaoduo
Wen, David Xu, Anand Natarajan, Pavan Kumar,
Shiva Shankar P, Ritesh Tijoriwala, Karan Asher, Hao
Wu, Aarti Basant, Daniel Ford, Delia David, Nezih Yigitbasi, Pratap Singh, and Carole-Jean Wu. Tectonic-Shift:
A composite storage fabric for Large-Scale ML training. In 2023 USENIX Annual Technical Conference
(USENIX ATC 23), pages 433–449, Boston, MA, July
2023. USENIX Association.

[77] Yuan Yu, Michael Isard, Dennis Fetterly, Mihai Budiu,
Úlfar Erlingsson, Pradeep Kumar Gunda, and Jon Currey. Dryadlinq: A system for general-purpose distributed
data-parallel computing using a high-level language. In
Proceedings of the 8th USENIX Conference on Operating Systems Design and Implementation, OSDI’08, page
1–14, USA, 2008. USENIX Association.
[78] Matei Zaharia, Mosharaf Chowdhury, Tathagata Das,
Ankur Dave, Justin Ma, Murphy McCauly, Michael J.
Franklin, Scott Shenker, and Ion Stoica. Resilient distributed datasets: A fault-tolerant abstraction for inmemory cluster computing. In 9th USENIX Symposium on Networked Systems Design and Implementation (NSDI 12), pages 15–28, San Jose, CA, April 2012.
USENIX Association.
[79] Matei Zaharia, Tathagata Das, Haoyuan Li, Scott
Shenker, and Ion Stoica. Discretized streams: An efficient and Fault-Tolerant model for stream processing on
large clusters. In 4th USENIX Workshop on Hot Topics
in Cloud Computing (HotCloud 12), Boston, MA, June
2012. USENIX Association.
[80] Hanyu Zhao, Zhenhua Han, Zhi Yang, Quanlu Zhang,
Mingxia Li, Fan Yang, Qianxi Zhang, Binyang Li,
Yuqing Yang, Lili Qiu, Lintao Zhang, and Lidong Zhou.
Silod: A co-design of caching and scheduling for deep
learning clusters. In Proceedings of the Eighteenth European Conference on Computer Systems, EuroSys ’23,
page 883–898, New York, NY, USA, 2023. Association
for Computing Machinery.
[81] Hanyu Zhao, Zhi Yang, Yu Cheng, Chao Tian, Shiru
Ren, Wencong Xiao, Man Yuan, Langshi Chen, Kaibo
Liu, Yang Zhang, Yong Li, and Wei Lin. Goldminer:
Elastic scaling of training data pre-processing pipelines
for deep learning. Proc. ACM Manag. Data, 1(2), jun
2023.
[82] Mark Zhao, Niket Agarwal, Aarti Basant, Buğra Gedik,
Satadru Pan, Mustafa Ozdal, Rakesh Komuravelli, Jerry
Pan, Tianshu Bao, Haowei Lu, Sundaram Narayanan,
Jack Langman, Kevin Wilfong, Harsha Rastogi, CaroleJean Wu, Christos Kozyrakis, and Parik Pol. Understanding data storage and ingestion for large-scale deep rec17

