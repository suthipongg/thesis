arXiv:2406.14571v1 [cs.AR] 11 Jun 2024

PreSto: An In-Storage Data Preprocessing System
for Training Recommendation Models
Yunjae Lee†

Hyeseong Kim†

Minsoo Rhu

School of Electrical Engineering
KAIST
yunjae408@kaist.ac.kr

School of Electrical Engineering
KAIST
hyeseong.kim@kaist.ac.kr

School of Electrical Engineering
KAIST
mrhu@kaist.ac.kr

Traditionally, the RecSys training pipeline employed offline
data preprocessing where the raw data retrieved from the
storage system is transformed into train-ready tensors in
advance and gets archived at a separate storage space for future
usage. However, the proliferation of petabyte-scale data and
the wide variety of RecSys models developed by ML engineers
have rendered offline preprocessing to incur an intractable
amount of overhead [70]. Specifically, it becomes increasingly
difficult to provision the substantial storage space required to
store all the data preprocessed offline, while also adapting to
changes in the newly developed RecSys models.
These challenges have triggered a shift towards online
preprocessing, which involves preprocessing the raw data “onthe-fly”. Online preprocessing obviates the need to separately
store the preprocessed data, so it helps better respond to
changes in the model architecture [70]. Nevertheless, these
online preprocessing approaches introduce several system-level
challenges, potentially causing a throughput mismatch between
I. I NTRODUCTION
data preprocessing and ML model training. Consider a scenario
Deep neural network (DNN) based machine learning (ML) al- where preprocessing and training jobs are co-located within
gorithms have demonstrated their effectiveness in a wide range the same GPU training server node, i.e., preprocessing is
of application domains. Among the successfully deployed ML undertaken on the host CPU that also manages the GPUapplications, recommendation systems (RecSys) have emerged side training job. If the CPU does not have a high enough
as a highly effective tool for online content recommendation computation power for preprocessing, it fails in generating
services. Such rising demand for recommendation services has sufficient amount of train-ready tensors that the GPU can
rendered hyperscalers to dedicate significant resources to the consume, leading to significant GPU underutilization (less than
development and training of diverse RecSys models to ensure 20% GPU utility, Section III-A). To address these challenges,
high-quality inference services. Unlike latency-optimized ML Zhao et al. [70] and Audibert et al. [5] suggest a server
inference, training algorithms are throughput-hungry workloads disaggregation solution where a pool of CPU servers is reserved
that favor high-performance, throughput-optimized accelerators for preprocessing purposes. Disaggregating CPU servers for
like GPUs. However, these power-hungry GPUs account preprocessing allows hundreds to thousands of CPU cores to be
for a large portion of ML system’s operating expenses, so allocated on-demand, even for a single data preprocessing job,
maintaining high GPU utilization becomes critical for lowering effectively closing the performance gap between preprocessing
TCO (total cost of ownership). Unfortunately, keeping the and model training thereby minimizing GPU idle time [25],
RecSys training pipeline busy with minimal GPU idle time [70]. However, this baseline “CPU-centric” disaggregated
requires the “data preprocessing” stage to preprocess an ample preprocessing incurs significant deployment cost and power
amount of raw data, so that the preprocessed, train-ready tensors consumption due to the large number of pooled CPU servers.
Given this landscape, an important objective of our work is
can be fed into the GPU in a seamless manner.
to characterize baseline CPU-centric disaggregated data prepro† Co-first authors who contributed equally to this research.
cessing systems targeting production-scale RecSys models, rootcausing its critical system-level challenges. A key observation
This is the author preprint version of the work. The authoritative version will
appear in the Proceedings of the 51st IEEE/ACM International Symposium we make is that the majority of data preprocessing time is
on Computer Architecture (ISCA-51), 2024.
spent conducting feature generation and feature normalization
Abstract—Training recommendation systems (RecSys) faces
several challenges as it requires the “data preprocessing” stage
to preprocess an ample amount of raw data and feed them to
the GPU for training in a seamless manner. To sustain high
training throughput, state-of-the-art solutions reserve a large
fleet of CPU servers for preprocessing which incurs substantial
deployment cost and power consumption. Our characterization
reveals that prior CPU-centric preprocessing is bottlenecked on
feature generation and feature normalization operations as it
fails to reap out the abundant inter-/intra-feature parallelism in
RecSys preprocessing. PreSto is a storage-centric preprocessing
system leveraging In-Storage Processing (ISP), which offloads the
bottlenecked preprocessing operations to our ISP units. We show
that PreSto outperforms the baseline CPU-centric system with a
9.6× speedup in end-to-end preprocessing time, 4.3× enhancement
in cost-efficiency, and 11.3× improvement in energy-efficiency on
average for production-scale RecSys preprocessing.
Index Terms—Recommendation system, computational storage
device, near data processing, neural network

operations, which inherently contain high inter-/intra-feature preprocessing includes Fourier transform and normalization of
parallelism. However, the latency-optimized CPU architectures, audio data [57]. Similarly, RecSys also requires a unique data
the de facto standard in data preprocessing, fail to fully exploit preprocessing to generate the train-ready tensors consumed by
the abundant inter-/intra-feature parallelism in feature gener- the model training stage, which we detail below.
ation and normalization, leading to sub-optimal performance.
Traditionally, the RecSys training pipeline employed offline
This in turn leads to the feature generation and normalization to data preprocessing where the raw feature data stored in the
account for 79% of the RecSys data preprocessing time, causing storage system is transformed into train-ready tensors well
the most significant performance bottleneck. To make up for the before model training takes place. However, the proliferation
meager preprocessing throughput provided with CPUs, the data of petabyte-scale data and the frequent development of dipreprocessing stage necessitates a large number of CPU cores verse RecSys models by ML engineers have made offline
(up to several hundreds) to be allocated so that its aggregate preprocessing impractical because it is difficult to manage
preprocessing throughput matches the throughput demands the substantial storage space required to store the offline
of GPU’s model training stage, which leads to substantial preprocessed data. This challenge has triggered a shift towards
deployment cost.
online preprocessing [70], which involves preprocessing the raw
In this work, we propose to employ accelerated computing feature data “on-the-fly” (Figure 1). The train-ready tensors
for RecSys data preprocessing to fundamentally address its are derived from the features that are constantly generated
system-level challenges at low cost. Because the abundant online by inference services. Specifically, inference servers log
inter-/intra-feature parallelism in data preprocessing is well- various end-user’s interactions with the inference service as
suited for domain-specific acceleration, our first key proposal distinct features (e.g., news feed a user has clicked, items a
is to offload the time-consuming feature generation and nor- user has purchased) using logging engines, e.g., Meta’s Scribe
malization operations to our accelerator for high-performance [28]. Additionally, various streaming and batch engines, such
data preprocessing. An important design decision still remains, as Spark [69], further label and filter data before storing them
however, regarding where our data preprocessing accelerator in a centralized data warehouse [72]. These raw feature data are
should be placed within the overall system architecture. State- categorized into two types: dense and sparse. Dense features
of-the-art data preprocessing solutions employ disaggregated represent continuous values (e.g., the time when a user viewed
CPU servers to dynamically allocate the right amount of a video from YouTube), while sparse features represent sparse
CPU cores for data preprocessing [5], [70]. While using our categorical values which can be variable-length (e.g., list of
data preprocessing accelerator as a drop-in replacement for YouTube videos a user has viewed over a one-hour period). The
CPUs within the disaggregated CPU servers is a practically logged features archived within the centralized data warehouse
feasible option, we observe that such a design point is sub- are then fetched into the storage system of each datacenter
optimal as it unnecessarily incurs high network traffic to for future data preprocessing. The data preprocessing stage,
copy data in (the raw data to be preprocessed) and out (the also known as the ETL (Extract, Transform, and Load) phase,
preprocessed train-ready tensors) of the disaggregated server involves the following series of operations:
for data preprocessing.
• (Extract) The logged raw feature data are first retrieved
To this end, we present PreSto (Preprocessing in-Storage),
from the storage system in preparation for the featurewhich is an In-Storage Processing (ISP) based data preprocessspecific transform operations.
ing system for RecSys training. In conventional systems, the
• (Transform) The extracted raw feature data go through
petabyte-scale raw data that is to be preprocessed are stored
feature generation and feature normalization in order to
in a distributed storage system. Rather than copying the raw
generate the train-ready tensors.
data over the datacenter network and preprocessing them at a
• (Load) The train-ready tensors are copied over to the
disaggregated, remote preprocessing server, PreSto conducts
GPU’s high-bandwidth memory (HBM) in preparation for
preprocessing near data using ISP. We demonstrate that such
the RecSys model training.
“storage-centric” data preprocessing can effectively close the
performance gap between preprocessing and model training
Once the train-ready tensors are loaded into GPU’s memory,
at a much lower cost compared to existing solutions. Overall, the actual model training is undertaken. Specifically, the GPU
PreSto provides high speedup on data preprocessing perfor- executes the embedding layers (embedding lookups, pooling for
mance (average 9.6×) at low cost, significantly reducing the embedding reductions), feature interactions (batched GEMM),
TCO (average 4.3×) and energy consumption (average 11.3×) and MLP layers (GEMM). When it comes to embedding layers,
vs. the baseline CPU-centric disaggregated preprocessing.
the embedding look-up operation retrieves embeddings from
an embedding table [51], which utilizes embedding indices
II. BACKGROUND
that are generated during the Transform phase of ETL (e.g.,
A. End-to-End RecSys Training Pipeline
the embedding indices transformed from feature X’ and W in
DNNs typically require some form of input data prepro- Figure 1 are utilized for embedding look-up operations). Recent
cessing before training. For instance, image classification work on various system-level performance optimizations for
requires preprocessing operations like image decoding, resizing, RecSys has primarily focused on this “training” stage of endcropping, or augmentation. Additionally, speech recognition to-end training pipeline, rather than data preprocessing. The

Inference
servers

Logs

Logging
engine

Datacenter
Storage system

Streaming/
Batch engine

User A,B,C

Extract

Feature X, W
for mini-batch 0

Data generation

Columnar format

Data storage

Extract

Feature X, W
for mini-batch 1

Dense
feature X

Bottom MLP

Load
Emb. indices
from feature X’

Train-ready tensors

User D,E,F
New
feature X’

…

Row-oriented format

Feature W

Mini-batch 0

Core 1

User D,E,F
Feature X
Formatting &
partitioning

New
feature X’

GPU

Mini-batch 1

Load

Emb. indices
from feature W

Top MLP

SSD 1

User
A,B,C
User
D,E,F

Feature W

Train-ready tensors

Pooling

Feature
X,Y,Z,W

User A,B,C
Feature X

Core 0

Inter-feature parallelism
3 Format
conversion

Feature interaction

Data warehouse

SSD 0

Intra-feature parallelism
1 Feature
2 Feature
generation
normalization

CPU

Embedding
table

Transformation

Data preprocessing

Model training

Fig. 1: High-level overview of the end-to-end RecSys training pipeline. In this work, we assume our baseline data storage and ingestion
pipeline for data preprocessing by referring to the related academic literature published by Meta [28], [70]–[72].

focus of this work is on RecSys data preprocessing, so we refer
to these relevant prior studies for more details on the RecSys
model architectures and training/inference [1], [4], [16]–[20],
[22], [29], [30], [34]–[37], [48], [51], [56], [61], [66].
While GPU-centric systems are popular options for training
purposes, it is worth emphasizing that the entire data preprocessing stage (the ETL phase) is executed using CPUs in
state-of-the-art data preprocessing systems [5], [14], [32], [47],
[49], [65], including those for RecSys [25], [50], [70], [71]. In
the rest of this paper, we assume such “CPU-centric” RecSys
data preprocessing system for our baseline system.
B. Key Properties of RecSys Data Preprocessing
While numerous prior work explored preprocessing for
vision, speech, and language processing [5], [8], [9], [14],
[32], [33], [47], [49], [53], [65], data preprocessing for RecSys
is relatively less explored [25], [50], [70], [71]. Unlike image or
audio data, RecSys data is represented in a tabular format with
multiple rows and columns. Concretely, each row represents
an individual “user” whereas each column represents a distinct
“feature” related to that user’s past interactions with the RecSys
inference service (shown as the data generation stage in
Figure 1).
In the context of online preprocessing, deciding which
features to utilize for model training depends on the ML
engineer’s choice, i.e., it is extremely challenging to predict
which specific features will be utilized. Consequently, the
hardware/software system for online preprocessing exhibits
some unique properties in the Extract and Transform phase:
• (Extract) The raw feature data is first converted and
stored in a columnar format (shown as the data storage
stage in Figure 1). A group of rows are sharded into
mutually exclusive partitions and different partitions are
stored as independent columnar files into a distributed
storage system of datacenter (e.g., the two partitions in
Figure 1 are stored as two separate columnar files over two
SSDs). The reason why these tabular data are converted
in a columnar format is to avoid overfetching unwanted
features. For example, in the data storage stage depicted
in Figure 1, with a columnar format, any given feature for
all the users can be selectively extracted from the storage

system without having to retrieve unwanted features, e.g.,
it is possible to only fetch features X and W without
having to fetch features Y and Z. In contrast, with the
original, row-oriented format, extracting features X and
W for all users inevitably leads to (unwanted) features Y
and Z to be retrieved, wasting data read bandwidth.
• (Transform) The transformations conducted at this phase
generate the mini-batch inputs that are utilized by the
GPU for model training. All transformation operations
performed within a mini-batch (detailed in the next subsection) are executed independently from transformations
targeting other mini-batches. This is because RecSys
transformation operations exhibit abundant inter-/intrafeature parallelism. Specifically, each element within a
given feature vector (e.g., user A and C’s feature X in
Figure 1) represents a given user’s interaction and there
exists no data dependency across different users. Therefore,
a transformation operation can be conducted element-wise
by exploiting intra-feature parallelism. Similarly, different
features (e.g., feature X and W for all users) are subject
to independent transformation operations by leveraging
inter-feature parallelism.
C. Feature Generation/Normalization in Data Preprocessing
RecSys data preprocessing can be divided into three key
steps. First, the feature generation step generates new features
using the raw feature data extracted from storage (Step ❶ in
Figure 1, e.g., a new feature X’ is generated from the raw
feature X). Notably, one of the representative sparse feature
generation operations is the “Bucketize” operation [63], [70],
which transforms dense features into sparse features by sharding
features based on predefined bucket boundaries (Algorithm 11 ).
Once the desired number of features is generated, they undergo
feature normalization (Step ❷). Common feature normalization
techniques include “Log” (which normalizes dense features
using a logarithmic function) and “SigridHash” [63], [70]
(which normalizes sparse features by computing a hash value
1 This paper focuses on the feature generation (Bucketize) and feature
normalization (SigridHash, Log) operations publicly available in the opensource TorchArrow [63]. Algorithm 1 and Algorithm 2 are simplified versions
of each algorithm implemented in TorchArrow.

GPU training node

1: Input dense feature a[1 . . . n]; bucket boundary b[1 . . . m];

CPU

CPU

CPU

PCIe

CPU

CPU

CPU

(a)

…

GPU … GPU

…

GPU

CPU

…

output c[1 . . . n]
2: /* Digitize input dense features based on bucket */
3: for i ← 1 to n do
4:
/* Find the index of buckets to which the input value
belongs using binary search algorithm*/
5:
c[i] ← SearchBucketID(a[i], b[1 . . . m])
6: end for

A pool of CPU servers

CPU

CPU

CPU

GPU training node
CPU

Network

Algorithm 1 Bucketize for feature generation [63]

PCIe
GPU

GPU … GPU

(b)

Fig. 2: System architectures for RecSys training. (a) A system that
co-locates CPU-based data preprocessing workers with GPU-based
model training workers within the same server node. (b) A system that
provisions a pool of disaggregated CPU servers for data preprocessing.

Algorithm 2 SigridHash for feature normalization [63]
1: Input sparse feature a[1 . . . n]; seed s; max value d; output

c[1 . . . n];
2: /* Apply hashing to input sparse features and limit their

values */
3: for i ← 1 to n do
4:
/* Compute seeded hash function */
5:
h ← ComputeHash(a[i], s)
6:
c[i] ← h mod d
7: end for

that maps those features within the maximum index of the
corresponding embedding table of the RecSys model, see
Algorithm 2). Finally, the normalized features are converted
into an input mini-batch (Step ❸), which eventually gets loaded
into GPU memory for training the RecSys model.
D. System Architecture for CPU-centric Data Preprocessing
Software architecture. As shown in Figure 1, the RecSys
training pipeline employs the producer-consumer model. GPU
training workers load and consume train-ready tensors (i.e.,
mini-batches) that the CPU preprocessing workers generate
by transforming raw feature data. Because all transformation
operations conducted within a mini-batch are executed locally
without any dependencies to other mini-batches, state-of-theart frameworks for end-to-end RecSys training pipeline such
as TorchRec [24] allocate a single worker per each CPU
core to handle the generation of each train-ready tensors
that constitute a given mini-batch. By spawning multiple
CPU workers in parallel, multiple input mini-batches are
concurrently generated, enabling scalable improvements in
data preprocessing throughput.
Hardware architecture. While the software architecture of
RecSys data preprocessing provides high scalability, a critical
challenge remains regarding how to allocate a sufficient number
of CPU cores that provide high enough data preprocessing
throughput that meets the throughput demands of GPU-side
training? Traditionally, model training and data preprocessing
jobs are co-located within the same server node containing
multiple GPUs (Figure 2(a)) [47], [49]. As such, the performance of such co-located training pipeline is limited by
how many CPU cores are available within the same server
node (e.g., NVIDIA DGX system contains 8 A100 GPUs
and 128 CPU cores, allowing a single GPU to utilize 16

CPU cores for data preprocessing), potentially suffering from
significant GPU underutilization when the aggregate CPUside data preprocessing throughput underwhelms the GPU-side
training performance (detailed in Section III-A).
To address these challenges, Zhao et al. [70] and Audibert
et al. [5] proposed server disaggregation where a pool of
CPU servers is reserved for data preprocessing (Figure 2(b)).
Disaggregating CPU servers allows a large pool of CPU
cores to be elastically allocated on-demand for preprocessing,
effectively closing the performance gap between preprocessing
and model training [25], [70]. However, such design point
incurs substantial deployment cost and power consumption due
to the large number of pooled CPU servers.
III. C HARACTERIZATION AND M OTIVATION
In this paper, we utilize the open-source RecSys data
preprocessing library TorchArrow [63] to conduct a workload
characterization study on state-of-the-art, CPU-centric data
preprocessing systems. We note that there exists a significant
disparity between the publicly available RecSys dataset [10]
and the characteristics of a production-level dataset mentioned
by a recent work from Meta [70]. Specifically, compared to
the public dataset, production-level RecSys datasets contain
a much larger number of dense/sparse features and larger
average sparse feature length. Such discrepancy can undermine
the primary objective of our study which is to characterize
state-of-the-art RecSys preprocessing. Consequently, we scale
up the open-sourced Criteo dataset [10] (referred to as RM1 in
this paper) and develop four synthetic RecSys models (RM2-5)
based on [70] to better represent the properties of productionlevel RecSys datasets. Table I summarizes the details of our
public/synthetic datasets and the RecSys models trained. Our
characterization is conducted with a training batch size of
8,192. Section V further details our evaluation methodology.
A. Motivation
As discussed in Section II-D, the aggregate data preprocessing throughput is strictly determined by how many CPU
cores (i.e., the number of data preprocessing workers) are
utilized for preprocessing. Consider a co-location based RecSys
training system (Figure 2(a)) using a state-of-the-art DGX
server [52] which contains 8 A100 GPUs and 128 CPU cores,
allowing a single GPU to utilize 16 CPU cores for data
preprocessing. In Figure 3, we scale up the number of CPU
cores for preprocessing (from 1 to a maximum of 16) and

Data preprocessing configuration parameters
Type
Public

# Dense feats. # Sparse feats.

RecSys model architecture

Avg. sparse # Generated
Bucket size Bottom MLP
feat. length sparse feats.

Top MLP

# Tables

Avg.
# Embeddings

RM1

13

26

1 (fixed)

13

1024

512-256-128 1024-1024-512-256-1

39

500,000

RM2
RM3
Synthetic
RM4
RM5

504
504
504
504

42
42
42
42

20
20
20
20

21
42
42
42

1024
1024
2048
4096

512-256-128
512-256-128
512-256-128
512-256-128

63
84
84
84

500,000
500,000
500,000
500,000

1024-1024-512-256-1
1024-1024-512-256-1
1024-1024-512-256-1
1024-1024-512-256-1

Throughput
(samples/s)

A100 utilization

100

Max training throughput

120,000

75

80,000

50

40,000

25

0

0
1

2
4
8
Number of CPU cores for preprocessing

16

Number of CPU cores

300
200
100
0
RM3

RM4

Extract (Decode)
Format conversion

Bucketize
Else

SigridHash
Load

10
5

RM1

400

RM2

15

Extract (Read)
Log

0

Fig. 3: Effective preprocessing throughput (left axis) and the resulting
GPU utilization (right axis) as a function of the number of CPU cores
(i.e., number of preprocessing workers) utilized for preprocessing.
The dotted line shows the upperbound, maximum training throughput
achievable using a single NVIDIA A100 GPU (left axis), which
assumes the GPU is seamlessly fed with sufficient amount of trainready tensors without interruption. To measure GPU’s utilization,
we use the CUDA Profiling Tools Interface (CUPTI) library. The
experiment is collected over the evaluation platform detailed in
Section V using our synthetic model RM5.

RM1

Preprocessing time
(normalized)

Data preprocessing

160,000

GPU utilization (%)

TABLE I: The RecSys training dataset configuration and the target model architecture. RM1 is based on the public Criteo dataset [10] and
RM2-5 are synthetically generated models we created by referring to production-grade RecSys dataset’s characteristics released by Meta [70].

RM5

Fig. 4: The number of CPU cores required for CPU-centric preprocessing to fully utilize a training node containing 8 A100 GPUs.

study its effect on data preprocessing throughput (left axis) and
the percentage of execution time the A100 GPU is actually
training the model (right axis). We make the following two
key observations from this experiment. First, the preprocessing
throughput increases almost linearly as a function of the number
of CPU cores, achieving 15× throughput improvement with 16
preprocessing workers vs. a single worker. Second, even with
16 preprocessing workers (i.e., the maximum number of CPU
cores that can be allocated in co-located RecSys preprocessing),
the GPU spends less than 20% of its execution time actually
conducting model training as the train-ready tensors are not
being sufficiently supplied to the GPU. Consequently, the
end-to-end training throughput gets bounded by the effective
preprocessing throughput which is well below the maximum
training throughput achievable (dotted line).
Server disaggregation for data preprocessing [5], [25], [70]
is an effective solution to close such wide performance gap, as
it enables the allocation of any number of CPU preprocessing
workers as required by the GPU training stage (Figure 2(b)). In
Figure 4, we derive the number of CPU cores required in the

RM2

RM3

RM4

RM5

Fig. 5: Latency to preprocess a single mini-batch input using a single
preprocessing worker in the baseline CPU-centric system, broken
into key steps of preprocessing. The “Extract” stage (Section II-A)
is further divided into (1) latency to fetch encoded raw feature data
from the remote storage node (denoted as “Extract (Read)”) and (2)
latency spent decoding them (denoted as “Extract (Decode)”). As
depicted, data preprocessing is bounded by the compute-intensive
feature generation and normalization operations, rather than I/O
operations (“Extract (Read)”). All results are normalized to RM1.

data preprocessing stage to sustain the model training stage’s
high throughput requirement. For production-level synthetic
datasets with large number of features and sparse feature
lengths, several hundreds of CPU cores are required (367
cores for RM5) to sufficiently supply the train-ready tensors
for an 8 A100 GPU server node. It is important to note that
hundreds to thousands of such production-level RecSys models
are developed by ML engineers, invoking numerous concurrent
training jobs executed over several tens of thousands of highperformance GPUs across the datacenter fleet [25], [70]. Such
high demand for RecSys model training directly translates into
substantial cost and power consumption in maintaining the
disaggregated CPU servers for data preprocessing, e.g., Meta
states that up to 60% of power consumption in RecSys training
pipeline is dedicated to the storage and data ingestion pipeline
of online CPU-centric data preprocessing [70].
Given such, a key motivation of this work is to conduct a
detailed characterization on CPU-centric RecSys data preprocessing systems. In the remainder of this section, we root-cause
the system-level bottlenecks of existing solutions in search for
a scalable and cost-effective preprocessing system for RecSys.
B. Breakdown of End-to-End Data Preprocessing Time
To identify the key bottlenecks in RecSys data preprocessing,
Figure 5 first evaluates end-to-end latency to preprocess a
single training mini-batch during the ETL phase. Compared
to the public RM1, the production-level RM2-5 contain a
larger number of dense and sparse features with larger average
sparse feature lengths. As such, these production-level models

RM1

Log

Bucketize SigridHash

100
80
60
40
20
0

GPU training node

LLC hit rate (%)

LLC hit rate

A pool of accelerators

GPU training node
CPU
CPU

Acc

Acc

Acc

PCIe

Acc

Acc

Acc

…

Bucketize SigridHash

CPU

…

Utilization (%)

Memory bandwidth

…

100
80
60
40
20
0

GPU … GPU

Acc

Acc

Acc

SSD

SSD … SSD

PCIe
Switch

Switch

GPU … GPU … GPU

Acc … Acc … Acc

GPU

Log

RM5

Fig. 6: CPU and memory bandwidth utilization (left) and LLC hit
rate (right) during the execution of feature generation/normalization
operations for the public model (RM1) and the production-scale model
(RM5). We utilize Linux perf for our evaluation.

Network

Storage system

SSD

(a)

SSD

Network

SSD … SSD

Storage system

SSD

(b)

Fig. 7: (a) Preprocessing accelerators co-located with GPUs and (b)
disaggregated pool of preprocessing accelerators.

experience a substantial increase in total preprocessing time
D. Our Goal: Scalable & Cost-Effective Preprocessing
where RM5 experiences the largest 14× increase in latency.
Overall, we conclude that current CPU-centric RecSys
Specifically, with the larger number of features in the RM2preprocessing
systems are bounded by the level of computation
5 models, both dense and sparse feature normalization time
power
available
with CPUs, failing to fully reap out the inter(i.e., Log and SigridHash, respectively) account for up to
/intra-feature
parallelism
inherent in preprocessing. Although
55% of its preprocessing time. Additionally, these productiondisaggregating
a
pool
of
CPUs for data preprocessing can
grade RM2-5 models experience a notable increase in feature
help
meet
the
computation
demands of preprocessing, it
generation time (i.e., Bucketize) due to the large number of
requires
high
deployment
cost
and
high power consumption. We
sparse features to generate and its large bucket size. While
argue
that
the
abundant
inter-/intra-feature
parallelism in data
RM3-5 all have the same number of sparse features to generate
preprocessing
is
well-suited
for
domain-specific
acceleration,
(at 42), larger bucket sizes (from 1024 to 4096 in RM3 to
proposing
an
accelerated
computing
system
for
RecSys
data
RM5, m in Algorithm 1) incur longer feature generation time
preprocessing
which
is
scalable
and
cost-effective.
because the latency to search the bucket ID of each feature
value increases. Conversely, despite RM1-3 all having the same
IV. PreSto: A N I N -S TORAGE “P RE ” PROCESSING
bucket size at 1024, the larger number of sparse features to
A RCHITECTURE FOR R EC S YS T RAINING
generate (from 13 to 42 in RM1 to RM3) leads to larger feature
We present PreSto (Preprocessing in-Storage), our Ingeneration time as well.
Storage Processing (ISP) based data preprocessing system
Overall, feature generation (Bucketize) and feature normaliza- for RecSys training. Our proposed system offloads the timetion (SigridHash and Log) collectively account for an average consuming feature generation and normalization operations to
79% of the data preprocessing time and become the most an accelerator that is tightly coupled with the storage system.
significant performance limiter in RecSys preprocessing.
A. System Design Considerations
C. Analysis on Performance-Limiting Operations
Figure 6 shows the CPU and memory bandwidth utilization
and the last-level cache (LLC) hit rate during the execution of
the three key performance-limiting operations (i.e., Bucketize,
SigridHash, Log) in RM1 and RM5, respectively. Our analysis
reveals the following key insights. First thing to note is that all
three operations exhibit high CPU utilization with relatively
low memory bandwidth utilization. RM5 in particular achieves
increased memory bandwidth utilization as it has more features
to generate and normalize compared to RM1. Nonetheless, the
memory bandwidth utility of RM5 still remains under 15% of
the maximum 281.6 GB/sec of memory throughput, exhibiting
a compute-bound behavior. While the feature generation and
normalization operations require large amount of data accesses,
the actual working set it touches upon is relatively small. In
RM1 and RM5, it amounts to as small as several tens of
KBs to at most tens of MBs. For instance, the Bucketize
operation involves sharding features based on the predefined
bucket range whose active working set fits well within on-chip
caches, leading to an LLC hit rate of 85%.

Our key proposition is to employ accelerated computing for
data preprocessing, so an important question to be answered is
where our accelerator(s) should be deployed within the system
architecture. Below we elaborate on the two possible design
points that retrofit our data preprocessing accelerator within
conventional co-located and disaggregated server architectures.
Preprocessing accelerators co-located with GPUs. Figure 7(a) illustrates our first design point, a scale-“up” server
architecture employing a PCIe switch to co-locate the preprocessing accelerators with the GPUs. When the number of
co-located accelerators is large enough to meet the GPU’s
training demands, this design point can obviate the need
for disaggregated CPU servers dedicated to preprocessing.
However, a critical limitation of such scale-up server is its
scalability because the aggregate preprocessing throughput
is still bounded by the total number of accelerators colocated within this scale-up server. As such, for large-scale
RecSys models whose data preprocessing demands exceed the
preprocessing throughput available with co-located accelerators,
the GPU training workers will still suffer from idle periods.
Another key concern with this approach is that both the
accelerator-side data preprocessing workers and the GPU-side

PCIe
GPU

SSD

GPU … GPU

FPGA

GPU training node

CPU

Network

FPGA

SSD

Data

GPU

CPU

FPGA

SSD

Control

GPU training node

…

PreSto ISP units

Storage system

Training job

7 Run

training

Framework
1 Train information

6 Transfer

Train manager

Model

mini-batch

Input queue
2 Initialization

Fig. 8: System architecture of PreSto.

B. Proposed Approach: In-Storage Data Preprocessing
Hardware architecture. Due to the aforementioned challenges of co-located and disaggregated accelerator systems, our
proposed system employs ISP (in-storage processing) architectures for data preprocessing, i.e., in-storage “pre”processing.
As shown in Figure 8, this approach utilizes ISP devices like
SmartSSDs [59] to directly replace conventional SSD cards. A
SmartSSD tightly couples a normal SSD with a lightweight
FPGA device within the NVMe U.2 form factor. This allows
SmartSSDs to become a drop-in replacement for normal SSDs
while still staying within the NVMe’s 25 Watts power envelope2 .
As such, a SmartSSD can utilize its local FPGA device to
implement our preprocessing accelerator right next to the local
SSD where the raw feature data is stored. Such design point
effectively tackles the system challenges of both co-located
and disaggregated preprocessing accelerators as follows.
1) Scalability. As discussed in Figure 1, the tabular raw
feature data subject for preprocessing is stored as columnar
files. A group of rows within the tabular data is sharded
into partitions and different partitions are stored as
independent columnar files in a distributed storage system
2 A high-end FPGA card like Xilinx U280 [67] which has a TDP of 225
Watts cannot be utilized for a U.2 SmartSSD card.

PreSto

training workers all time-share the PCIe bus to communicate
with the host CPU. Because preprocessing workers and training
workers concurrently execute in a training pipeline, the PCIe
bus can become a performance hotspot under such unbalanced
system architecture. Given such critical limitations, we conclude
that such scale-up solution is impractical.
Disaggregated pool of preprocessing accelerators. To
address the scalability issue in our scale-up server design,
an alternative solution would be to utilize our preprocessing
accelerator as a drop-in replacement of CPUs in the disaggregated preprocessing CPU pool. As shown in Figure 7(b),
this design point effectively provides a disaggregated pool
of preprocessing accelerators to the GPU training workers.
By decoupling training jobs from preprocessing jobs over
distinct server pools, the optimal number of preprocessing
accelerators to allocate that meets a target training job’s
throughput demands can be determined dynamically, providing
high scalability. Furthermore, this design point can better
exploit inter-/intra-feature parallelism using domain-specific
acceleration for high efficiency. However, similar to the baseline
CPU-centric disaggregated preprocessing, server disaggregation
still incurs substantial deployment cost and high TCO.

Network

CPU

5 Preprocessed

data

Preprocess manager

SmartSSD
ISP

3 Start
preprocessing

4 P2P
+ Preprocessing

Fig. 9: Major components of PreSto software system and key steps
undertaken during end-to-end RecSys training.

(e.g., 2 columnar files stored in two SSDs in Figure 1).
While multiple blocks that constitute a single columnar
file can further be distributed across multiple storage
devices, state-of-the-art file systems for RecSys (e.g.,
Meta’s Tectonic file system [55]) store all the blocks
in a given partition contiguously within a single storage
device. This ensures that all preprocessing operations
can be conducted locally within a SmartSSD because
all transformation operations conducted within a minibatch (i.e., partition) are executed locally without any
dependencies to other mini-batches (i.e., partitions in
other columnar files). Consequently, in our proposed
system, the overall preprocessing throughput can scale
proportionally with the number of SmartSSDs allocated
to a preprocessing worker targeting a given training job.
A training job with a target preprocessing throughput in
mind is first allocated with the appropriate number of
SmartSSDs (detailed later in this subsection), one that is
large enough to sustain the training stage’s preprocessing
need. We then have each SmartSSD independently extract
raw feature data from its local SSD, which is immediately
P2P transferred over to the local FPGA device for onthe-fly preprocessing. Because multiple SmartSSDs (i.e.,
multiple preprocessing workers) concurrently conduct
preprocessing and generate mini-batch inputs, our PreSto
ISP units provide highly scalable preprocessing service.
2) Efficiency. In our proposed system, all preprocessing
operations are conducted locally within the storage system.
This is because the SmartSSD’s FPGA accelerator can extract the raw feature data directly from its local SSD using
P2P data transfers, obviating the need for a disaggregated
accelerator server pool with high maintenance cost. Such
design decision also helps eliminate the communication
overhead associated with disaggregated accelerator server
designs (Figure 7(b)), which requires the raw feature
data to be transferred from the storage system to the
remote accelerator pool. It is also worth pointing out that
SmartSSDs can seamlessly be deployed within the power
constraints of the baseline distributed storage system, all
thanks to the use of commodity devices that operate within

On-chip
feature
buffer
Global
memory
(DRAM)

Bucketize unit

Bucket buffer

On-chip
feature
buffer

SigridHash
unit

On-chip
feature
buffer

Log unit

Decoder
unit
Feature
generation unit

SSD

PCIe switch (Peer-to-Peer)

Software architecture. Figure 9 provides a high-level
overview of our software architecture. The two key components
of our software system are the train manager and the preprocess
manager. The train manager is implemented as part of the
training worker process whose main role is to manage the endto-end model training job, from data preprocessing to model
training. That is, it requests the mini-batch inputs (i.e., trainready tensors) from the storage system and once the mini-batch
inputs are returned, they are forwarded to the GPU for model
training. The preprocess manager is in charge of spawning and
managing the actual preprocessing workers using the SmartSSD
devices. Once the mini-batch inputs are ready, the preprocess
manager returns them back to the train manager. Below we
summarize the major steps undertaken during the end-to-end
RecSys training process.

Decoder

Feature
normalization unit

the NVMe SSD’s power budget (less than 25 watts per
device [6]). Consequently, PreSto is minimally intrusive to
existing hardware infrastructure while maintaining powerefficiency via accelerated computing.

Fig. 10: PreSto accelerator microarchitecture.

data from the local SSD and transfers them P2P to the
FPGA to conduct preprocessing on-the-fly (step ❹).
4) Once the mini-batch inputs are ready, they are converted
into a train-ready tensor format, as required by TorchRec,
and copied over to the input queue within the train
manager (step ❺).
5) Finally, the train manager transfers the mini-batch from its
input queue to the GPU and kicks off the model training
process (step ❻ and ❼). Because multiple SmartSSDs are
concurrently preprocessing raw features and replenishing
the train manager’s input queue, it ensures that the GPU is
constantly supplied with sufficient amount of mini-batch
inputs to consume.

1) When a training job is launched by TorchRec [24],
the train manager receives important information about
the target training job (e.g., model configuration for
training/preprocessing, mini-batch size, and other metadata) and goes through several boot-strapping procedures
in preparation for model training. These include the
allocation of an input queue to store the mini-batch inputs
designated for model training and a Remote Procedure C. Data Preprocessing Accelerator Microarchitecture
The design objective of our PreSto accelerator is to maxiCall (RPC) initialization (step ❶ in Figure 9).
2) Using the training job information, the train manager mally exploit inter/intra-feature parallelism inherent in RecSys
measures the maximum training throughput achievable data preprocessing. As depicted in Figure 10, we use the
with GPUs for the given training job. This is done by custom logic within the FPGA to implement a hardwired
supplying the GPUs with dummy mini-batch inputs and decoder unit, feature generation units, and feature normalization
stress-testing their highest sustainable throughput. This units. Each unit is equipped with the essential hardware
process only takes tens of seconds, the overhead of which logic tailored to the following operations: “Decoder” for
is amortized over the several hours/days worth of training columnar file decoding (our columnar files assume the Apache
time. The train manager then initializes the preprocess Parquet file format [3]), “Bucketize” for feature generation, and
manager, sending information relevant to the preprocessing “SigridHash” as well as “Log” for feature normalization. To
jobs (e.g., configuration parameters of preprocessing). maximally exploit inter-/intra-feature parallelism, we employ
One of the important information that is forwarded to the following design optimizations. First, to exploit inter-feature
the preprocess manager is GPU’s maximum training parallelism, we deploy multiple processing elements dedicated
throughput (T ), as it determines the level of preprocessing to each individual feature, directly connected to the off-chipthroughput that must be sustained in order to fully utilize memory interface to fully utilize the bandwidth of global
the GPUs for model training. As such, the preprocess memory (DRAM). Second, to exploit intra-feature parallelism,
manager measures offline the maximum preprocessing each processing element employs double-buffering to overlap
throughput delivered with a single SmartSSD device under the next feature value’s data fetch operation with the current
the given preprocessing configuration (P). By dividing feature value’s generation and normalization operations. That
the maximum training throughput with the per-SmartSSD is, once a portion of an input feature is fetched on-chip, its
preprocessing throughput (T /P), the preprocess manager transformation is immediately executed while the next feature
derives the number of SmartSSD devices that need to be value is concurrently being fetched from the off-chip memory.
allocated to fully saturate the GPUs (step ❷).
V. M ETHODOLOGY
3) The preprocess manager launches the necessary number
of preprocessing workers based on the derived number of A. Benchmarks
SmartSSD (T /P) (step ❸). As each preprocessing worker
The majority of current academic research on RecSys utilizes
independently generates mini-batch inputs locally within the Criteo dataset [10], which is the largest publicly available
the SmartSSD, each device fetches its share of raw feature dataset for RecSys. The Criteo dataset consists of 13 dense

Unit

LUT

REG

BRAM URAM

DSP

Decode

18.84%

8.49%

25.08%

0.00%

is designed using Xilinx Vitis HLS 2022.2 whose resource
utilization is summarized in Table II.
As for the analytical model we developed for large-scale
Bucketize 7.88% 4.28% 6.19% 27.59% 0.00%
performance
estimations, we utilize the observations made
SigridHash 23.11% 12.47% 11.89% 0.00% 19.19%
from our characterization study in Section III where data
Log
4.18% 2.79% 4.89% 0.00% 10.62%
preprocessing operations are embarrassingly parallel and exhibit
Total
54.02% 28.03% 48.05% 27.59% 29.81%
high scalability. Because end-to-end training performance as
well
as its data preprocessing performance is throughput-bound
TABLE II: FPGA resource utilization of PreSto’s preprocessing
accelerator. Decode, Bucketize, SigridHash, and Log units are rather than latency-bound, our analytical performance model
synthesized with an operating frequency of 223MHz.
assumes that the preprocessing throughput measured from our
real PoC prototype (i.e., the preprocessing throughput measured
and 26 sparse features, with a fixed feature length of 1 for with a single CPU core (baseline) and a single SmartSSD
each sparse feature. According to recent work from Meta [70], (PreSto)) scales proportionally with the number of CPU cores
production-level RecSys models can amount to 504 dense and allocated (baseline) or the number of SmartSSDs allocated
42 sparse features with an average sparse feature length of (PreSto) for data preprocessing.
Software. Our end-to-end RecSys training pipeline is imple20, much larger than the public Criteo dataset. To narrow this
gap in our evaluation, we additionally construct four synthetic mented using TorchArrow (v0.1.0) [63] for data preprocessing
RecSys models in accordance with [70], expanding the existing and TorchRec (v0.3.2) [24] for model training, assuming a
features of the Criteo dataset to better cover the evaluation mini-batch size of 8,192. We assume the Apache Parquet file
space of production-level RecSys models with large number format [3] when the columnar raw feature data is stored in
of dense/sparse features. Table I details the configuration of our storage system. Both baseline CPU-centric and PreSto
the five RecSys models and their training datasets we evaluate. preprocessing system communicate with the GPU training
node using the PyTorch RPC API [11]. We use Xilinx Runtime
library to manage PreSto’s FPGA device.
B. Experimental Setup
0.00%

Hardware. Exploring PreSto in a production-level training C. Evaluation Methods
pipeline that accurately reflects industry’s large-scale disagPower measurement. When measuring the power congregated CPU servers, multi-node/multi-GPU systems, and a sumption of the CPU-based storage node as well as the
distributed storage array integrated with SmartSSDs is challeng- disaggregated preprocessing nodes, we measure its systeming at the academic research level for several reasons. Aside level power consumption using Intel Performance Counter
from many undisclosed details of hyperscaler’s production ML Monitor (PCM) [23]. The Xilinx Vivado [68] and NVIDIA
infrastructure, the unavailability of SmartSSDs in cloud services System Management Interface (nvidia-smi) are used when
like Amazon AWS [2] rendered our experiments to employ measuring the power of PreSto’s FPGA accelerator and the
the following methodology. We first demonstrate PreSto’s GPU, respectively.
advantages in real systems by constructing a small-scale,
Cost-efficiency. To quantify the cost-effectiveness of PreSto,
proof-of-concept (PoC) prototype of PreSto using commodity we also evaluate cost-efficiency using the evaluation metric
CPU/GPU/SmartSSD devices. We then develop an analytical suggested in [42], [43] as summarized below:
model that estimates PreSto’s performance at large-scale by
T hroughput × Duration
utilizing real measurements from our PoC prototype.
Cost-efficiency =
CapEx + OpEx
At a high-level, our PoC prototype includes three major
components: (1) a storage node (with and without a SmartSSD
where OpEx = ∑(Power × Duration × Electricity)
to model PreSto (with SmartSSD) and baseline storage system
(without SmartSSD)), (2) a GPU training node, and (3) a pool
CapEx ($) refers to the one-time capital expenditure required
of multiple CPU nodes for preprocessing (to model baseline to purchase and establish the hardware platform components.
disaggregated CPU preprocessing service), which communicate OpEx ($) represents the operating expenditure of this hardware
over a network using 10 Gbps Ethernet. Both the storage node platform. To determine CapEx, we utilize the cost information
and the pool of CPU nodes for preprocessing are designed using obtained from the respective company website [12], [59].
a total of three two-socket Intel Xeon Gold 6242 CPU nodes OpEx is derived using the power consumed by the hardware
(32 CPU cores per node) where one node is used as the storage components (Power), the active duration of each hardware
node and the other two nodes are utilized as a remote pool for component (Duration, a period of 3 years [7], [43]), and the
data preprocessing (maximum 2×32=64 CPU cores available average price of electricity (Electricity, $0.0733/kWh [42],
for data preprocessing). The GPU training node contains AMD [43]). It is worth pointing out that the numerator value to
EPYC 7502 CPU connected with a single NVIDIA A100 GPU. calculate cost-efficiency (T hroughput×Duration) is identical
When evaluating PreSto, we add a single SmartSSD card [59] for both baseline disaggregated CPU preprocessing and PreSto:
to the storage node which handles data preprocessing locally both baseline and our proposal can sustain the throughput
within the storage node. PreSto’s preprocessing accelerator demands of GPU’s training stage, so T hroughput and Duration

Disagg(1)
Disagg(16)
Disagg(32)
Disagg(64)
PreSto

Disagg(1)
Disagg(16)
Disagg(32)
Disagg(64)
PreSto

RM3

RM4

RM5

Fig. 11: Preprocessing throughput of PreSto (single SmartSSD) vs.
Disagg. Disagg(N) is a design point executing with N preprocessing
workers using N CPU cores. Results are normalized to Disagg(1).
SigridHash
Load

Log
Speedup

RM1

RM2

RM3

RM4

PreSto

Disagg

PreSto

Disagg

PreSto

Disagg

PreSto

Disagg

PreSto

18
15
12
9
6
3
0

Speedup

Bucketize
Else

1.2
1
0.8
0.6
0.4
0.2
0

Disagg

Latency
(normalized)

Extract
Format conversion

PreSto

3
2
1
0
RM2

RM3

RM4

RM5

Fig. 13: Aggregate latency incurred during any RPC calls executed
for inter-node communication during the course of data preprocessing.
PreSto

12

Disagg

400

9

300

6

200

3

100

0

0
RM1

RM2

RM3

RM4

RM5

Number of CPU cores

Disagg(1)
Disagg(16)
Disagg(32)
Disagg(64)
PreSto

RM2

Network overhead
(normalized)

Disagg(1)
Disagg(16)
Disagg(32)
Disagg(64)
PreSto

RM1

Disagg

4

RM1

Number of ISP units

Disagg(1)
Disagg(16)
Disagg(32)
Disagg(64)
PreSto

Throughput
(normalized)

70
60
50
40
30
20
10
0

Fig. 14: The number of ISP units (left axis) and CPU cores (right
axis) required for PreSto and Disagg to sustain a single multi-GPU
server node containing 8 A100 GPUs.

RM5

Fig. 12: (Left axis) Data preprocessing time to generate a single minibatch using a single preprocessing worker. Latency is broken down
into key steps of data preprocessing. (Right axis) PreSto’s end-to-end
speedup for preprocessing. All results are normalized to Disagg.

the encoded raw feature data from local SSD to the FPGA
which is immediately followed by their decoding using our
dedicated decoder unit. Because the decoding algorithm is
less parallelizable than feature generation and normalization
are constant values. Therefore, the difference in cost-efficiency operations, the reduction in the “Extract” step’s execution time
is less pronounced, rendering this step to account for an average
is determined by (CapEx+OpEx).
40.8% of the total preprocessing time of PreSto. Nonetheless,
VI. E VALUATION
PreSto provides significant improvements in the performance of
We first demonstrate PreSto’s merits using our PoC pro- feature generation (Bucketize) and normalization (SigridHash,
totype (Section VI-A). We then utilize our analytical model Log), achieving an average 9.6× (maximum 11.6×) reduction
(Section V) to estimate PreSto’s effect on performance, energy- in end-to-end preprocessing time. These results highlight the
efficiency, and TCO at larger scale (Section VI-B). In the benefits of PreSto’s domain-specific acceleration using RecSys
rest of this section, the baseline CPU-centric preprocessing preprocessing’s inter-/intra-feature parallelism.
system assumes the disaggregated CPU server design (denoted
Data movements. Another key benefit provided with PreSto
“Disagg”), one which is equipped with a maximum of 64 CPU is that all data preprocessing operations are conducted locally
cores in our small-scale PoC prototype.
within the storage node, unlike Disagg which needs to explicitly
copy data in (the raw data to be preprocessed) and out
A. Performance and Cost-Effectiveness of PreSto (PoC)
(the train-ready tensors) of the disaggregated CPU nodes for
Throughput. Figure 11 compares the data preprocessing data preprocessing. Because our small-scale PoC prototype
throughput of PreSto vs. Disagg. As depicted, a single evaluates a single training job in a highly controlled, isolated
SmartSSD device (PreSto) consistently outperforms Disagg setting, Disagg’s RPC communication time to read out the raw
even with 32 CPU cores (i.e., a single CPU node) and feature data from the remote storage node and copy into the
demonstrates the benefits of our ISP solution. Since Disagg’s disaggregated CPU nodes accounts for a relatively small portion
performance scales well to the number of CPU cores (workers) of the end-to-end preprocessing time (but still accounting for
utilized, allocating more CPU cores can still match the 9.1% in RM2 under Disagg in Figure 12). Since real-world
throughput provided with PreSto albeit at a proportional datacenter fleets concurrently handle a large number of training
increase in cost (e.g, Disagg(64) using two CPU nodes (64 jobs, all of which time-share the datacenter network, PreSto’s
cores) is able to slightly outperform PreSto by average 27% ISP capability can be beneficial in alleviating the preprocessing
but with 2× higher cost).
operation’s pressure on network communications. In Figure 13,
Latency. To better highlight where PreSto’s speedup comes we show the aggregate latency incurred during any RPC calls
from, Figure 12 compares the latency to generate a single executed for inter-node data movements during the course
mini-batch input using a single preprocessing worker using of data preprocessing. Unlike Disagg which incurs additional
Disagg and PreSto. The latency breakdown focuses on the key latency when the preprocessing worker copies raw feature data
steps undertaken during preprocessing. In the baseline Disagg, from the remote storage to the disaggregated CPU nodes, our
the “Extract” step includes time to fetch encoded raw feature PreSto can completely eliminate such performance overhead.
data from the remote storage node and decode them. With This leads PreSto to provide a 2.9× reduction in RPC-invoked
PreSto, the “Extract” step includes the P2P data transfer of inter-node communication time.

RM4

RM5

1x

(b)

A100
U280
PresSto (U280)
PreSto (SmartSSD)

A100
U280
PresSto (U280)
PreSto (SmartSSD)

A100
U280
PresSto (U280)
PreSto (SmartSSD)

RM1

RM2

RM3

RM4

RM5

Throughput/power
(normalized)

15
12
9
6
3
0

Throughput/power

A100
U280
PresSto (U280)
PreSto (SmartSSD)

5
4
3
2
1
0

A100
U280
PresSto (U280)
PreSto (SmartSSD)

Throughput (normalized)

Fig. 15: (a) Energy-efficiency and (b) cost-efficiency. Power consumption is measured using Intel’s Performance Counter Monitor (PCM)
and Xilinx Vivado. Section V-C details our methodology.

Fig. 16: Data preprocessing’s (left axis) performance and (right
axis) performance/Watt over PreSto (single SmartSSD device), PreSto
(single U280 FPGA), a single A100 GPU and a single U280 FPGA.

B. PreSto’s Effect on Energy-Efficiency and TCO
We now evaluate PreSto’s effect on performance/Watt
(energy-efficiency) and performance/$ (TCO) by utilizing our
analytical model for large-scale experiments (Section V-B).
Energy-efficiency. As discussed in Figure 4, baseline Disagg
requires significant number of CPU cores for data preprocessing
(e.g., 367 cores for RM5), which translates into significant
power consumption and high deployment cost. To quantitatively demonstrate PreSto’s effectiveness in energy and cost
reduction, we evaluate how many ISP units (i.e., the number
of SmartSSD cards) are required to match the preprocessing
demands of a multi-GPU server containing 8 GPUs (Figure 14).
Remarkably, to match such high GPU training throughput
demand, PreSto only requires a maximum of 9 ISP units which
incur (9×25)=225 Watts of worst-case power consumption
(25 Watts TDP per each SmartSSD card). Disagg, on the other
hand, requires up to 367 CPU cores (i.e., 12 CPU server
nodes) to match PreSto’s preprocessing performance, incurring
much higher power consumption as well as cost. Figure 15(a)
summarizes how all of this translate into energy consumption.
Overall, PreSto provides an average 11.3× (maximum 15.1×)
energy-efficiency improvement, demonstrating its merits.
Cost-efficiency (TCO). Figure 15(b) compares the costefficiency of PreSto and Disagg for data preprocessing (as
defined in Section V-C). Overall, PreSto provides an average
4.3× (maximum 5.6×) improvement in cost-efficiency vs.
Disagg. Cost-efficiency is primarily determined by both CapEx
and OpEx and our experiments thus far have demonstrated that
PreSto outperforms Disagg on both fronts, providing significant
reduction in TCO.
C. PreSto vs. Alternative Accelerated Preprocessing
Our study thus far have focused on comparing data preprocessing over the baseline disaggregated CPU servers and

2x

4x

1x

Bucketize

2x

4x

1x

SigridHash

Diasgg
Presto

RM3

Diasgg
Presto

RM2

Diasgg
Presto

RM1

(a)

Diasgg
Presto

0

Diasgg
Presto

RM1 RM2 RM3 RM4 RM5

2

Diasgg
Presto

0

2x

4x

Speedup

4

300
250
200
150
100
50
0

Speedup

Diasgg
Presto

5

6

600
500
400
300
200
100
0

Diasgg
Presto

10

PreSto

Diasgg
Presto

15

Disagg

8

Latency (normalized)

PreSto
Cost-efficiency
(normalized)

Energy-efficiency
(normalized)

Disagg

20

Log

Fig. 17: Latency of Disagg and PreSto’s feature generation/normalization time (left axis) and PreSto’s speedup (right axis)
when the the number of features for preprocessing is changed. The
“1×” data point corresponds to the RM5 configuration in Table I.
Disagg’s latency to conduct each operation is normalized to its
respective “1×” latency of PreSto.

ISP architectures. For the completeness of our study, we also
evaluate the efficacy of alternative accelerated preprocessing solutions where high-end GPUs/FPGAs function as preprocessing
accelerators. Figure 16 compares the preprocessing performance
(left axis) and performance/Watt (energy-efficiency) with four
system design points: (1) a single A100 GPU (denoted “A100”)
and (2) a single Xilinx U280 [67] FPGA (denoted “U280”)
employed within a disaggregated accelerator pool (Figure 7(b)),
as well as our proposed FPGA-based accelerator system
implemented using (3) a single, discrete U280 FPGA card
integrated within the storage node over PCIe (denoted “PreSto
(U280)”) and (4) one implemented using a single SmartSSD
device (denoted “PreSto (SmartSSD)”). We utilizes NVIDIA’s
NVTabular library [54] for GPU-based data preprocessing. The
U280-based FPGA accelerator is synthesized with 2× larger
number of Decoder, Feature generation, and Feature normalization units that maximally utilize U280’s larger custom logics.
PreSto (SmartSSD) provides an average 2.5× speedup vs.
A100 while experiencing an average 5% performance loss vs.
U280 FPGA. Note that PreSto (SmartSSD) is able to achieve
such performance despite its much lower power consumption
(TDP of 25 Watts (SmartSSD) vs. 250 Watts (A100) and 225
Watts (U280)). In general, GPUs are throughput-optimized
devices with high compute and memory throughput, so they
perform best when the target application requires massive
compute and memory accesses. We observe that the compute
and memory operations entailed in RecSys preprocessing is
lightweight vs. training. This makes it challenging for the
GPU to amortize the cost of CUDA kernel launches, each of
which has a small working set with modest compute/memory
operations, leading to significant GPU underutilization. U280
does much better than the GPU but its end-to-end speedup
vs. PreSto (SmartSSD) is relatively low, due to its high
latency overhead in copying data in/out of the disaggregated
preprocessing node (which accounts for an average 47.6% of its
end-to-end preprocessing time). Even though PreSto (U280)
minimizes such redundant data movements and achieves a
slightly higher preprocessing throughput compared to PreSto
(SmartSSD), PreSto (SmartSSD) delivers much higher energyefficiency (an average 2.9×) vs. PreSto (U280) by being
custom-designed to right-size its compute units for data
preprocessing under a tighter power budget (25 Watts).

preprocessing, which is as discussed in this paper completely
We investigate PreSto’s sensitivity to the number of features orthogonal operation compared to model training/inference.
Domain-specific/general-purpose ISP designs. There is a
to preprocess by evaluating the latency incurred in executing the
large
body of prior literature exploring domain-specific/generalBucketize, SigridHash, and Log operations when the number
purpose
ISP designs. GLIST [39] proposed an ISP architecture
of generated, sparse, and dense features are changed. Figure 17
for
SSD-based
GNN inference. SmartSAGE [38] proposed
illustrates a comparison of the average latency to execute
an
ISP-based
GNN
training system to overcome I/O bottleeach operation using Disagg and PreSto. While the latency
neck
of
SSD-based
training.
Mahapatra et al. [43] proposed
of Disagg increases almost proportionally with the number
an
ASIC-based
ISP
in
a
disaggregated
storage system for
of features for preprocessing, PreSto does a much better
serverless
functions,
alleviating
communication
overhead of
job leveraging inter-/intra-feature parallelism and consistently
remote
storage
systems.
RecSSD
[66]
and
RM-SSD
[61]
achieves significant speedups, demonstrating the robustness of
proposed
ISP
to
overcome
the
memory
bottlenecks
of
RecSys
our proposal.
inference. GraphSSD [46] proposed an ISP architecture for
graph semantics with a simple programming interface. ECSSD
VII. R ELATED WORK
[40] proposed an ISP architecture for extreme classification
There is a large body of prior literature exploring DNN based on the approximate screening algorithm. Work by Hu
preprocessing, RecSys model training/inference, RecSys data et al. [21] proposed an ISP-based dynamic multi-resolution
storage and preprocessing, and domain-specific/general-purpose storage system to mitigate the performance bottleneck of
ISP designs, which we summarize below.
data preparation for approximate compute kernels. ASSASIN
DNN preprocessing. There are multiple prior work address- [73], INSPIRE [41], and GenStore [45] proposed an ISP
ing the performance gap between DNN model training and architecture for stream computing, private information retrieval,
data preprocessing [5], [8], [9], [13], [14], [32], [33], [47], and genome sequence analysis, respectively. There also exists
[49], [53], [57], [65]. Mohan et al. [47] and Murray et al. [49] a large body of prior literature targeting ISP acceleration
proposed software optimizations to address this problem, while for data-intensive workloads. Morpheus [64], DeepStore [44],
TrainBox [57] and DALI [53] proposed hardware accelerated Active Flash [62], GraFBoost [27], Biscuit [15], and BlueDBM
preprocessing tackling computer vision and audio training tasks. [26] proposed a domain-specific ISP architecture that targets
Similarly, DLBooster [9] focused on offloading preprocessing data analytics, data management, object (de)serialization, or
operations to an FPGA for inference. PreGNN [13] proposed graph analytics. Summarizer [31] and INSIDER [58] proposed
offloading graph neural network (GNN) preprocessing tasks a hardware/software co-designed ISP architecture to offload
to an accelerator. Several prior art [5], [14], [32], [33], [65] data-intensive tasks with a set of flexible programming APIs.
proposed disaggregated preprocessing solutions but these works Willow [60] proposed architectural support to enhance the
primarily focused on efficiently managing CPU resources effectiveness and flexibility of a programmable ISP design
via software optimizations using data caching or prefetching. targeting I/O-intensive applications. Unlike these prior work,
Importantly, all of these prior art strictly do not focus on PreSto demonstrates the merits of an ISP solution targeting
RecSys, rendering the key contribution of our work unique.
compute-bound RecSys data preprocessing and uncovers its
RecSys data storage and preprocessing. Zhao et al. [70] new system-level bottlenecks, rendering our key contributions
discusses a disaggregated data preprocessing service for Meta’s unique.
RecSys training pipeline. XDL [25] proposed a distributed ML
VIII. C ONCLUSION
framework for Alibaba’s production RecSys model. InTune [50]
presents a reinforcement learning-based RecSys data pipeline
In this work, we propose an ISP based RecSys data preprooptimization to efficiently manage CPU resources. There also cessing system called PreSto which conducts the preprocessing
exists prior work exploring feature deduplication to improve operation close to where the training samples are preserved.
the performance of RecSys data preprocessing [71]. While not By fully leveraging inter-/intra-feature parallelism available in
targeting the preprocessing stage of RecSys, Tectonic-shift [72] feature generation/normalization, PreSto can effectively close
explored the viability of a flash storage tier in the data storage the performance gap between preprocessing and model training
system of Meta’s production ML training infrastructure, aiming at a much lower cost and power consumption compared to
to improve the performance and power efficiency of their I/O the baseline CPU-centric system. Overall, PreSto outperforms
operation in data storage stage. Overall, the contribution of state-of-the-art preprocessing systems with 9.6× speedup in
PreSto is orthogonal to these studies.
end-to-end preprocessing time, 4.3× improvement in costRecSys model training/inference. The surge of interest in efficiency, and 11.3× enhancement in energy-efficiency.
RecSys in both academia and industry has spawned numerous
ACKNOWLEDGEMENTS
prior work accelerating RecSys training and inference utilizing
near-/in-memory processing [4], [29], [34], [35], [56] as well as
This work was supported by the National Research Foundavarious hardware/software optimizations [1], [16]–[20], [22], tion of Korea (NRF) grant funded by the Korea government
[30], [36], [37], [48], [51]. Importantly, PreSto stands apart (MSIT) (NRF-2021R1A2C2091753), and Institute of Informafrom this body of work by focusing on accelerating RecSys tion & communications Technology Planning & Evaluation
D. PreSto Sensitivity to the Number of Features to Preprocess

(IITP) grant funded by the Korea government(MSIT) (No.
2022-0-01037, Development of High Performance Processingin-Memory Technology based on DRAM). We also appreciate
the support from SNU-SK Hynix Solution Research Center
(S3RC) and the EDA tool supported by the IC Design Education
Center(IDEC), Korea. Minsoo Rhu is the corresponding author.
R EFERENCES
[1] B. Acun, M. Murphy, X. Wang, J. Nie, C.-J. Wu, and K. Hazelwood,
“Understanding Training Efficiency of Deep Learning Recommendation
Models at Scale,” in Proceedings of the International Symposium on
High-Performance Computer Architecture (HPCA), 2021.
[2] Amazon, “Amazon AWS Cloud Computing Services,” https://aws.amazon.
com/, 2023.
[3] Apache Software Foundation, “Apache Parquet,” https://parquet.apache.
org/, 2023.
[4] B. Asgari, R. Hadidi, J. Cao, D. E. Shim, S.-K. Lim, and H. Kim,
“FAFNIR: Accelerating Sparse Gathering by Using Efficient NearMemory Intelligent Reduction,” in Proceedings of the International
Symposium on High-Performance Computer Architecture (HPCA), 2021.
[5] A. Audibert, Y. Chen, D. Graur, A. Klimovic, J. Šimša, and C. A.
Thekkath, “tf.data service: A Case for Disaggregating ML Input Data
Processing,” in Proceedings of the ACM Symposium on Cloud Computing
(SoCC), 2023.
[6] A. Barbalace and J. Do, “Computational Storage: Where Are We Today?”
in 11th Annual Conference on Innovative Data Systems Research (CIDR),
2021.
[7] L. A. Barroso, U. Hölzle, and P. Ranganathan, The Datacenter as
a Computer: Designing Warehouse-Scale Machines, Third Edition.
Springer Nature, 2019.
[8] W. Chen, S. He, Y. Xu, X. Zhang, S. Yang, S. Hu, X.-H. Sun, and G. Chen,
“iCache: An Importance-Sampling-Informed Cache for Accelerating I/OBound DNN Model Training,” in Proceedings of the International
Symposium on High-Performance Computer Architecture (HPCA), 2023.
[9] Y. Cheng, D. Li, Z. Guo, B. Jiang, J. Lin, X. Fan, J. Geng, X. Yu, W. Bai,
L. Qu, R. Shu, P. Cheng, Y. Xiong, and J. Wu, “DLBooster: Boosting Endto-End Deep Learning Workflows with Offloading Data Preprocessing
Pipelines,” in Proceedings of the 48th International Conference on
Parallel Processing (ICPP), 2019.
[10] Criteo, “Criteo Terabyte Click Logs,” https://labs.criteo.com/2013/12/
download-terabyte-click-logs/, 2013.
[11] P. Damania, S. Li, A. Desmaison, A. Azzolini, B. Vaughan, E. Yang,
G. Chanan, G. J. Chen, H. Jia, H. Huang, J. Spisak, L. Wehrstedt,
L. Hosseini, M. Krishnan, O. Salpekar, P. Belevich, R. Varma, S. Gera,
W. Liang, S. Xu, S. Chintala, C. He, A. Ziashahabi, S. Avestimehr,
A. Jain, and Z. DeVito, “PyTorch RPC: Distributed Deep Learning
Built on Tensor-Optimized Remote Procedure Calls,” in Proceedings of
Machine Learning and Systems (MLSys), 2023.
[12] Dell, “Dell R640,” https://www.dell.com/en-us/dt/servers/poweredgerack-servers.htm.
[13] D. Gouk, S. Kang, M. Kwon, J. Jang, H. Choi, S. Lee, and M. Jung,
“PreGNN: Hardware Acceleration to Take Preprocessing Off the Critical
Path in Graph Neural Networks,” in IEEE Computer Architecture Letters,
2022.
[14] D. Graur, D. Aymon, D. Kluser, T. Albrici, C. A. Thekkath, and
A. Klimovic, “Cachew: Machine Learning Input Data Processing as
a Service,” in Proceedings of USENIX Conference on Annual Technical
Conference (ATC), 2022.
[15] B. Gu, A. S. Yoon, D.-H. Bae, I. Jo, J. Lee, J. Yoon, J.-U. Kang, M. Kwon,
C. Yoon, S. Cho, J. Jeong, and D. Chang, “Biscuit: A Framework for
Near-Data Processing of Big Data Workloads,” in Proceedings of the
International Symposium on Computer Architecture (ISCA), 2016.
[16] U. Gupta, S. Hsia, V. Saraph, X. Wang, B. Reagen, G.-Y. Wei, H.-H. S.
Lee, D. Brooks, and C.-J. Wu, “DeepRecSys: A System for Optimizing
End-to-End At-Scale Neural Recommendation Inference,” in Proceedings
of the International Symposium on Computer Architecture (ISCA), 2020.
[17] U. Gupta, S. Hsia, J. Zhang, M. Wilkening, J. Pombra, H.-H. S. Lee, G.Y. Wei, C.-J. Wu, and D. Brooks, “RecPipe: Co-Designing Models and
Hardware to Jointly Optimize Recommendation Quality and Performance,”
in Proceedings of the International Symposium on Microarchitecture
(MICRO), 2021.

[18] U. Gupta, C.-J. Wu, X. Wang, M. Naumov, B. Reagen, D. Brooks,
B. Cottel, K. Hazelwood, M. Hempstead, B. Jia, H.-H. S. Lee,
A. Malevich, D. Mudigere, M. Smelyanskiy, L. Xiong, and X. Zhang,
“The Architectural Implications of Facebook’s DNN-Based Personalized
Recommendation,” in Proceedings of the International Symposium on
High-Performance Computer Architecture (HPCA), 2020.
[19] K. Hazelwood, S. Bird, D. Brooks, S. Chintala, U. Diril, D. Dzhulgakov,
M. Fawzy, B. Jia, Y. Jia, A. Kalro, J. Law, K. Lee, J. Lu, P. Noordhuis,
M. Smelyanskiy, L. Xiong, and X. Wang, “Applied Machine Learning at
Facebook: A Datacenter Infrastructure Perspective,” in Proceedings of the
International Symposium on High-Performance Computer Architecture
(HPCA), 2018.
[20] S. Hsia, U. Gupta, B. Acun, N. Ardalani, P. Zhong, G.-Y. Wei, D. Brooks,
and C.-J. Wu, “MP-Rec: Hardware-Software Co-design to Enable MultiPath Recommendation,” in Proceedings of the International Conference
on Architectural Support for Programming Languages and Operating
Systems (ASPLOS), 2023.
[21] Y.-C. Hu, M. T. Lokhandwala, T. I, and H.-W. Tseng, “Dynamic MultiResolution Data Storage,” in Proceedings of the International Symposium
on Microarchitecture (MICRO), 2019.
[22] R. Hwang, T. Kim, Y. Kwon, and M. Rhu, “Centaur: A Chiplet-Based,
Hybrid Sparse-Dense Accelerator for Personalized Recommendations,”
in Proceedings of the International Symposium on Computer Architecture
(ISCA), 2020.
[23] Intel, “Intel Performance Counter Monitor,” https://github.com/intel/pcm,
2022.
[24] D. Ivchenko, D. Van Der Staay, C. Taylor, X. Liu, W. Feng, R. Kindi,
A. Sudarshan, and S. Sefati, “TorchRec: a PyTorch Domain Library for
Recommendation Systems,” in Proceedings of the ACM Conference on
Recommender Systems (RecSys), 2022.
[25] B. Jiang, C. Deng, H. Yi, Z. Hu, G. Zhou, Y. Zheng, S. Huang, X. Guo,
D. Wang, Y. Song, L. Zhao, Z. Wang, P. Sun, Y. Zhang, D. Zhang,
J. Li, J. Xu, X. Zhu, and K. Gai, “XDL: An Industrial Deep Learning
Framework for High-dimensional Sparse Data,” in Proceedings of the 1st
International Workshop on Deep Learning Practice for High-Dimensional
Sparse Data, 2019.
[26] S.-W. Jun, M. Liu, S. Lee, J. Hicks, J. Ankcorn, M. King, S. Xu,
and Arvind, “BlueDBM: An Appliance for Big Data Analytics,” in
Proceedings of the International Symposium on Computer Architecture
(ISCA), 2015.
[27] S.-W. Jun, A. Wright, S. Zhang, S. Xu, and Arvind, “GraFBoost: Using
Accelerated Flash Storage for External Graph Analytics,” in Proceedings
of the International Symposium on Computer Architecture (ISCA), 2018.
[28] M. Karpathiotakis, D. Wernli, and M. Stojanovic, “Scribe: Transporting
Petabytes per Hour via a Distributed, Buffered Queueing System,” https:
//engineering.fb.com/2019/10/07/data-infrastructure/scribe/, 2019.
[29] L. Ke, U. Gupta, B. Y. Cho, D. Brooks, V. Chandra, U. Diril,
A. Firoozshahian, K. Hazelwood, B. Jia, H.-H. S. Lee, M. Li, B. Maher,
D. Mudigere, M. Naumov, M. Schatz, M. Smelyanskiy, X. Wang,
B. Reagen, C.-J. Wu, M. Hempstead, and X. Zhang, “RecNMP: Accelerating Personalized Recommendation with Near-Memory Processing,” in
Proceedings of the International Symposium on Computer Architecture
(ISCA), 2020.
[30] L. Ke, U. Gupta, M. Hempstead, C.-J. Wu, H.-H. S. Lee, and X. Zhang,
“Hercules: Heterogeneity-Aware Inference Serving for At-Scale Personalized Recommendation,” in Proceedings of the International Symposium
on High-Performance Computer Architecture (HPCA), 2022.
[31] G. Koo, K. K. Matam, T. I., H. K. G. Narra, J. Li, H.-W. Tseng,
S. Swanson, and M. Annavaram, “Summarizer: Trading Communication
with Computing Near Storage,” in Proceedings of the International
Symposium on Microarchitecture (MICRO), 2017.
[32] M. Kuchnik, A. Klimovic, J. Simsa, V. Smith, and G. Amvrosiadis,
“Plumber: Diagnosing and Removing Performance Bottlenecks in Machine
Learning Data Pipelines,” in Proceedings of Machine Learning and
Systems (MLSys), 2022.
[33] A. V. Kumar and M. Sivathanu, “Quiver: An Informed Storage Cache
for Deep Learning,” in Proceedings of USENIX Conference on File and
Storage Technologies (FAST), 2020.
[34] Y. Kwon, Y. Lee, and M. Rhu, “TensorDIMM: A Practical Near-Memory
Processing Architecture for Embeddings and Tensor Operations in
Deep Learning,” in Proceedings of the International Symposium on
Microarchitecture (MICRO), 2019.
[35] Y. Kwon, Y. Lee, and M. Rhu, “Tensor Casting: Co-Designing AlgorithmArchitecture for Personalized Recommendation Training,” in Proceedings

of the International Symposium on High-Performance Computer Architecture (HPCA), 2021.
[36] Y. Kwon and M. Rhu, “Training Personalized Recommendation Systems
from (GPU) Scratch: Look Forward not Backwards,” in Proceedings of
the International Symposium on Computer Architecture (ISCA), 2022.
[37] Y. Lee, S. H. Seo, H. Choi, H. U. Sul, S. Kim, J. W. Lee, and T. J. Ham,
“MERCI: Efficient Embedding Reduction on Commodity Hardware via
Sub-Query Memoization,” in Proceedings of the International Conference
on Architectural Support for Programming Languages and Operating
Systems (ASPLOS), 2021.
[38] Y. Lee, J. Chung, and M. Rhu, “SmartSAGE: Training Large-scale
Graph Neural Networks using In-Storage Processing Architectures,” in
Proceedings of the International Symposium on Computer Architecture
(ISCA), 2022.
[39] C. Li, Y. Wang, C. Liu, S. Liang, H. Li, and X. Li, “GLIST: Towards
In-Storage Graph Learning,” in Proceedings of USENIX Conference on
Annual Technical Conference (ATC), 2021.
[40] S. Li, F. Tu, L. Liu, J. Lin, Z. Wang, Y. Kang, Y. Ding, and Y. Xie,
“ECSSD: Hardware/Data Layout Co-Designed In-Storage-Computing Architecture for Extreme Classification,” in Proceedings of the International
Symposium on Computer Architecture (ISCA), 2023.
[41] J. Lin, L. Liang, Z. Qu, I. Ahmad, L. Liu, F. Tu, T. Gupta, Y. Ding,
and Y. Xie, “INSPIRE: IN-Storage Private Information REtrieval via
Protocol and Architecture Co-design,” in Proceedings of the International
Symposium on Computer Architecture (ISCA), 2022.
[42] M. Liu, S. Peter, A. Krishnamurthy, and P. M. Phothilimthana, “E3:
Energy-Efficient Microservices on SmartNIC-Accelerated Servers,” in
Proceedings of USENIX Conference on Annual Technical Conference
(ATC), 2019.
[43] R. Mahapatra, S. Ghodrati, B. H. Ahn, S. Kinzer, S. ting Wang,
H. Xu, L. Karthikeyan, H. Sharma, A. Yazdanbakhsh, M. Alian, and
H. Esmaeilzadeh, “Domain-Specific Computational Storage for Serverless
Computing,” arXiv preprint arXiv:2303.03483, 2023.
[44] V. S. Mailthody, Z. Qureshi, W. Liang, Z. Feng, S. G. de Gonzalo, Y. Li,
H. Franke, J. Xiong, J. Huang, and W.-m. Hwu, “DeepStore: In-Storage
Acceleration for Intelligent Queries,” in Proceedings of the International
Symposium on Microarchitecture (MICRO), 2019.
[45] N. Mansouri Ghiasi, J. Park, H. Mustafa, J. Kim, A. Olgun, A. Gollwitzer,
D. Senol Cali, C. Firtina, H. Mao, N. Almadhoun Alserr, R. Ausavarungnirun, N. Vijaykumar, M. Alser, and O. Mutlu, “GenStore: A HighPerformance In-Storage Processing System for Genome Sequence Analysis,” in Proceedings of the International Conference on Architectural
Support for Programming Languages and Operating Systems (ASPLOS),
2022.
[46] K. K. Matam, G. Koo, H. Zha, H.-W. Tseng, and M. Annavaram,
“GraphSSD: Graph Semantics Aware SSD,” in Proceedings of the
International Symposium on Computer Architecture (ISCA), 2019.
[47] J. Mohan, A. Phanishayee, A. Raniwala, and V. Chidambaram, “Analyzing
and Mitigating Data Stalls in DNN Training,” in Proceedings of the
VLDB Endowment (PVLDB), 2021.
[48] D. Mudigere, Y. Hao, J. Huang, Z. Jia, A. Tulloch, S. Sridharan, X. Liu,
M. Ozdal, J. Nie, J. Park, L. Luo, J. A. Yang, L. Gao, D. Ivchenko,
A. Basant, Y. Hu, J. Yang, E. K. Ardestani, X. Wang, R. Komuravelli,
C.-H. Chu, S. Yilmaz, H. Li, J. Qian, Z. Feng, Y. Ma, J. Yang,
E. Wen, H. Li, L. Yang, C. Sun, W. Zhao, D. Melts, K. Dhulipala,
K. Kishore, T. Graf, A. Eisenman, K. K. Matam, A. Gangidi, G. J.
Chen, M. Krishnan, A. Nayak, K. Nair, B. Muthiah, M. khorashadi,
P. Bhattacharya, P. Lapukhov, M. Naumov, A. Mathews, L. Qiao,
M. Smelyanskiy, B. Jia, and V. Rao, “Software-Hardware Co-design for
Fast and Scalable Training of Deep Learning Recommendation Models,”
in Proceedings of the International Symposium on Computer Architecture
(ISCA), 2022.
[49] D. G. Murray, J. Simsa, A. Klimovic, and I. Indyk, “tf.data: A Machine
Learning Data Processing Framework,” in Proceedings of the VLDB
Endowment (PVLDB), 2021.
[50] K. Nagrecha, L. Liu, P. Delgado, and P. Padmanabhan, “InTune:
Reinforcement Learning-Based Data Pipeline Optimization for Deep
Recommendation Models,” in Proceedings of the ACM Conference on
Recommender Systems (RecSys), 2023.
[51] M. Naumov, D. Mudigere, H.-J. M. Shi, J. Huang, N. Sundaraman,
J. Park, X. Wang, U. Gupta, C.-J. Wu, A. G. Azzolini, D. Dzhulgakov,
A. Mallevich, I. Cherniavskii, Y. Lu, R. Krishnamoorthi, A. Yu,
V. Kondratenko, S. Pereira, X. Chen, W. Chen, V. Rao, B. Jia,
L. Xiong, and M. Smelyanskiy, “Deep Learning Recommendation

Model for Personalization and Recommendation Systems,” arXiv preprint
arXiv:1906.00091, 2019.
[52] NVIDIA, “NVIDIA DGX A100,” https://images.nvidia.com/aem-dam/
Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf.
[53] NVIDIA, “NVIDIA Data Loading Library,” https://developer.nvidia.com/
dali, 2023.
[54] NVTabular, https://github.com/NVIDIA-Merlin/NVTabular, 2023.
[55] S. Pan, T. Stavrinos, Y. Zhang, A. Sikaria, P. Zakharov, A. Sharma,
M. Shuey, R. Wareing, M. Gangapuram, G. Cao, C. Preseau, P. Singh,
K. Patiejunas, J. Tipton, E. Katz-Bassett, and W. Lloyd, “Facebook’s
Tectonic Filesystem: Efficiency from Exascale,” in Proceedings of
USENIX Conference on File and Storage Technologies (FAST), 2021.
[56] J. Park, B. Kim, S. Yun, E. Lee, M. Rhu, and J. H. Ahn, “TRiM:
Enhancing Processor-Memory Interfaces with Scalable Tensor Reduction in Memory,” in Proceedings of the International Symposium on
Microarchitecture (MICRO), 2021.
[57] P. Park, H. Jeong, and J. Kim, “TrainBox: An Extreme-Scale Neural
Network Training Server Architecture by Systematically Balancing
Operations,” in Proceedings of the International Symposium on Microarchitecture (MICRO), 2020.
[58] Z. Ruan, T. He, and J. Cong, “INSIDER: Designing In-Storage Computing
System for Emerging High-Performance Drive,” in Proceedings of
USENIX Conference on Annual Technical Conference (ATC), 2019.
[59] Samsung, “Samsung SmartSSD,” https://samsungsemiconductor-us.com/
smartssd/.
[60] S. Seshadri, M. Gahagan, S. Bhaskaran, T. Bunker, A. De, Y. Jin, Y. Liu,
and S. Swanson, “Willow: A User-Programmable SSD,” in Proceedings
of USENIX Symposium on Operating Systems Design and Implementation
(OSDI), 2014.
[61] X. Sun, H. Wan, Q. Li, C.-L. Yang, T.-W. Kuo, and C. J. Xue, “RMSSD: In-Storage Computing for Large-Scale Recommendation Inference,”
in Proceedings of the International Symposium on High-Performance
Computer Architecture (HPCA), 2022.
[62] D. Tiwari, S. Boboila, S. Vazhkudai, Y. Kim, X. Ma, P. Desnoyers,
and Y. Solihin, “Active Flash: Towards Energy-Efficient, In-Situ Data
Analytics on Extreme-Scale Machines,” in Proceedings of USENIX
Conference on File and Storage Technologies (FAST), 2013.
[63] TorchArrow, https://github.com/pytorch/torcharrow, 2022.
[64] H.-W. Tseng, Q. Zhao, Y. Zhou, M. Gahagan, and S. Swanson,
“Morpheus: Creating Application Objects Efficiently for Heterogeneous
Computing,” in Proceedings of the International Symposium on Computer
Architecture (ISCA), 2016.
[65] T. Um, B. Oh, B. Seo, M. Kweun, G. Kim, and W.-Y. Lee, “FastFlow:
Accelerating Deep Learning Model Training with Smart Offloading of
Input Data Pipeline,” in Proceedings of the VLDB Endowment (PVLDB),
2023.
[66] M. Wilkening, U. Gupta, S. Hsia, C. Trippel, C.-J. Wu, D. Brooks,
and G.-Y. Wei, “RecSSD: Near Data Processing for Solid State Drive
Based Recommendation Inference,” in Proceedings of the International
Conference on Architectural Support for Programming Languages and
Operating Systems (ASPLOS), 2021.
[67] Xilinx, “Alveo U280 Data Center Accelerator Card,” https://www.xilinx.
com/products/boards-and-kits/alveo/u280.html.
[68] Xilinx, “Xilinx Vitis,” https://www.xilinx.com/products/design-tools/vitis.
html/, 2022.
[69] M. Zaharia, M. Chowdhury, T. Das, A. Dave, J. Ma, M. McCauly, M. J.
Franklin, S. Shenker, and I. Stoica, “Resilient Distributed Datasets: A
Fault-Tolerant Abstraction for In-Memory Cluster Computing,” in 9th
USENIX Symposium on Networked Systems Design and Implementation
(NSDI), 2012.
[70] M. Zhao, N. Agarwal, A. Basant, B. Gedik, S. Pan, M. Ozdal, R. Komuravelli, J. Pan, T. Bao, H. Lu, S. Narayanan, J. Langman, K. Wilfong,
H. Rastogi, C.-J. Wu, C. Kozyrakis, and P. Pol, “Understanding Data
Storage and Ingestion for Large-Scale Deep Recommendation Model
Training,” in Proceedings of the International Symposium on Computer
Architecture (ISCA), 2022.
[71] M. Zhao, D. Choudhary, D. Tyagi, A. Somani, M. Kaplan, S.-H. Lin,
S. Pumma, J. Park, A. Basant, N. Agarwal, C.-J. Wu, and C. Kozyrakis,
“RecD: Deduplication for End-to-End Deep Learning Recommendation
Model Training Infrastructure,” in Proceedings of Machine Learning and
Systems (MLSys), 2023.
[72] M. Zhao, S. Pan, N. Agarwal, Z. Wen, D. Xu, A. Natarajan, P. Kumar,
S. S. P, R. Tijoriwala, K. Asher, H. Wu, A. Basant, D. Ford, D. David,
N. Yigitbasi, P. Singh, and C.-J. Wu, “Tectonic-Shift: A Composite

Storage Fabric for Large-Scale ML Training,” in Proceedings of USENIX
Conference on Annual Technical Conference (ATC), 2023.
[73] C. Zou and A. A. Chien, “ASSASIN: Architecture Support for Stream
Computing to Accelerate Computational Storage,” in Proceedings of the
International Symposium on Microarchitecture (MICRO), 2022.

