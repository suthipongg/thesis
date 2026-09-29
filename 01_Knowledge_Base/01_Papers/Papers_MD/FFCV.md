FFCV: Accelerating Training by Removing Data Bottlenecks

arXiv:2306.12517v1 [cs.LG] 21 Jun 2023

Guillaume Leclerc*
leclerc@mit.edu
MIT

Andrew Ilyas*
ailyas@mit.edu
MIT

Logan Engstrom*
engstrom@mit.edu
MIT

Hadi Salman
hady@mit.edu
MIT

Sung Min Park
sp765@mit.edu
MIT

Aleksander Madry
madry@mit.edu
MIT

Abstract
We present FFCV, a library for easy and fast machine learning model training. FFCV speeds
up model training by eliminating (often subtle) data bottlenecks from the training process.
In particular, we combine techniques such as an efficient file storage format, caching, data
pre-loading, asynchronous data transfer, and just-in-time compilation to (a) make data loading
and transfer significantly more efficient, ensuring that GPUs can reach full utilization; and (b)
offload as much data processing as possible to the CPU asynchronously, freeing GPU cycles
for training. Using FFCV, we train ResNet-18 and ResNet-50 on the ImageNet dataset with
competitive tradeoff between accuracy and training time. For example, we are able to train an
ImageNet ResNet-50 model to 75% in only 20 mins on a single machine. We demonstrate FFCV’s
performance, ease-of-use, extensibility, and ability to adapt to resource constraints through
several case studies. Detailed installation instructions, documentation, and Slack support
channel are available at https://ffcv.io/.

1

Introduction

What is the limiting factor in faster model training? Hint: it is not always the GPUs. When training
a machine learning model, the life cycle of an individual example spans three stages: reading the
example into memory, processing the example in memory, and finally updating model parameters
with the example on GPU (e.g. by calculating and then following the gradient). The stage with the
lowest throughput determines the overall learning system’s throughput.
Our investigations (and others’ [MPR+21]) show that in practice the limiting factor is often not
computing updates, but rather the data reading and data processing stages. Indeed, in standard
training setups, the GPUs can spend a majority of cycles just waiting for inputs to process!
To better saturate GPUs and thereby increase training throughput, we present FFCV, a system
designed to reduce data loading and processing bottlenecks while remaining simple to use. FFCV
operates in two successive stages: preprocessing and train-time loading. In the first stage, FFCV
preprocesses the dataset into a format more amenable to high-throughput loading. Then, in the
train-time loading stage, FFCV’s data loader replaces the original learning system’s data loader
without requiring any other implementation changes.
* Equal contribution.

1

Together, FFCV data preprocessing and the FFCV data loader can drastically increase training
speeds without any learning algorithm modifications. To demonstrate, we train machine learning models for a number of tasks much faster than previous general purpose data loaders (e.g.,
PyTorch’s default data loaders) can support. While FFCV improves performance on most GPUs,
its effect is most pronounced on faster GPUs, which require higher throughput data loading to
saturate available compute capacity. We expect FFCV will only increase in utility as new GPUs
become faster.
Contributions. We introduce FFCV, a drop-in, general purpose training system for high throughput data loading. Using FFCV requires no algorithmic changes, and involves a nearly identical
API to standard data loading systems (e.g., the default PyTorch [PGM+19] data loader). FFCV
automatically handles the necessary data transfer, memory management, and data conversion
work that users usually manually optimize (or leave to suboptimal defaults). FFCV also replaces
the default data preprocessing and augmentation pipeline with one that is more efficient due to
(a) just-in-time compilation to machine code and (b) highly optimized memory management. We
explore potential use cases of FFCV and find dramatic speed-ups:
• ImageNet Training. We greatly improve ImageNet single node training throughput, achieving
competitive speed-accuracy trade-offs (Figure 1).
• Bootstrapping and grid search. We enable faster large-scale grid search by supporting samemachine, different-GPU training without any throughput penalty.
• Network filesystem-based training. Especially in cloud computing environments where network file systems are commonplace, data reading can greatly bottleneck learning systems. FFCV
enables faster data loading in a realistic read-constrained environment.
• Tasks beyond computer vision. We demonstrate FFCV’s ability to speed up almost any data
loading task by using it as a drop-in replacement to the default PyTorch data loader in a
GPU-enabled sparse regression solver.

ImageNet: ResNet50 8xA100 GPU

ImageNet: ResNet18 1xA100 GPU

0.79
0.78

TIMM A3
NVIDIA PyTorch

0.77
0.76

PyTorch
Example

0.75

FFCV
MosaicML Frontier

0.74
0

1

2

3

Time (hrs)

4

5

FFCV

0.71
0.70
0.69
0.68
0.67
0.66
6

PyTorch

0

2

4

30

Time (hrs)

32

34

Figure 1: Accuracy vs. training time when training a ResNet-50 on 8 A100s. FFCV achieves
competitive accuracy/training time trade-offs compared to standard baselines. As an example, we
can train ImageNet to 75% accuracy in less than 20 minutes on a single machine. In the plot on
the left, the red dots correspond to the Pareto frontier of models achievable with the MosaicML
fast training solution as of February 2022—since then, MosaicML has integrated FFCV into their
training pipelines.

2

FFCV
Idealized Training
ImageFolder Full Training
ImageFolder Reading + Processing
ImageFolder Data Reading Only
0

200

400

600

800

1000 1200

Time Per Epoch (sec)

Figure 2: Time taken across stages in ImageNet training (median over three runs). ImageFolder
refers to the default PyTorch data loader used to load ImageNet. We find that data loading (in
particular, data processing) is the major bottleneck of standard training. The idealized training
time, or the training time we would obtain with perfect data loading, is almost 30 times smaller
than the time required to just process all the training images. In the top column, FFCV achieves
nearly ideal training time by removing the data reading and processing bottlenecks.

2

Identifying Bottlenecks in Training

What makes a machine learning training system “slow” or “fast?” The answer varies by task,
algorithm, implementation, and computing equipment available at train time. Model training is
best thought of as a pipeline of discrete steps: data reading, data processing, and GPU computing
that executes the learning algorithm.
To understand which of these steps bottlenecks training in practice, we study a standard task
commonly used to benchmark training speeds [CNK+17; MCC+20], namely ImageNet [DDS+09]
training. As a specific setup, we investigate the PyTorch ImageNet training example with the
standard PyTorch ImageNet data loader, running on a standard AWS instance for GPU-based
learning: p4d.24xlarge machines, which have 8 A100 GPUs, 96 vCPUs, and enough RAM to fit the
ImageNet training set into memory. We benchmark each part of the system’s throughput; Figure 2
shows our results. Overall, we find that data loading bottlenecks this standard training setup, and,
furthermore, by fixing data loading we could achieve 30 times faster model training. Below we
explore this data loading bottleneck in further detail.
Data reading throughput. We begin by only benchmarking data read throughput, measuring
how long the data loader takes to read the entire dataset without performing any processing. As
the machine we test on can cache the entire ImageNet dataset into memory, the data reading step is
not a bottleneck (cf. Figure 2): it takes only 75 seconds.
Data processing throughput. To check whether data processing is a bottleneck, we measure
how long the data loader takes to read the entire dataset while also performing processing: JPEG
decoding, random cropping/resizing, random flipping, and normalization. We find that processing
is a major bottleneck: adding processing to reading greatly increases loading time to 1200 seconds
from the 70 seconds that loading alone took (see Figure 2).
Full training throughput. Finally, we measure the entire system’s throughput, including the
learning stage. We find that adding the learning stage does not greatly change the time taken to
iterate through the whole dataset (cf. Figure 2). The fact that our throughput does not decrease
despite adding learning on the GPUs indicates that the data loading/processing subsystem cannot

3

supply data fast enough to saturate the GPUs. Indeed, as further corroboration, we simulate how
fast model throughput could be by benchmarking our training process on a fixed data vector (here,
we require no data loading). Our throughput in this idealized setting, shown in the same figure
above, is much higher, and shows that we could train up to 30x faster with optimal data loading.

3

Eliminating Data Bottlenecks

Now, our focus turns to the question: how can we design a better data loading system? To maximize
performance, FFCV manages the entire data management pipeline, from the file format used to store
the training data all the way to data augmentations used at training time. Focusing on one step of
the data loading pipeline at a time, we show how FFCV’s implementation circumvents issues in
existing solutions to efficiently load data.

3.1

Challenge #1: Storing a Machine Learning Dataset

To eliminate data bottlenecks in the machine learning pipeline we start with the data format. There
are already multiple existing file formats designed to store machine learning datasets: the most
common of these formats (and indeed, the default one in PyTorch) is the file-based format, where
one stores each example as an individual file. In the context of image recognition, for example,
one saves each example as its own (typically JPEG-compressed) image file, and uses the enclosing
folder to encode the label. The file-based approach has some advantages—most notably, users can
intuitively interact with the examples on an individual level (e.g., they can open any training image
in a standard image viewer). However, this format is not at all optimized for performance, and
comes with several fundamental drawbacks.
FFCV introduces its own new file format: the .beton file. In the following, we discuss the
different considerations involved in designing this file format, and show that FFCV’s new file format
circumvents issues both with file-based formats as well as existing specialized solutions (namely,
WebDataset, TFRecord, and MXNet RecordIO).
Reducing filesystem strain. To reduce filesystem strain, existing specialized file formats either
group data examples in shards (WebDataset) or concatenate all the data into a single file (TFRecord,
RecordIO). FFCV adopts the latter option, but goes even further—by organizing the dataset into
pages (at the cost of some wasted space), it eliminates random read penalties by making it easy to
read data in large chunks. FFCV (along with TFRecord and RecordIO) datasets are also easier to
share than sharded data formats (e.g., WebDataset), since one only needs to transport a single file.
Flexibility. A data format should to be flexible enough to accommodate a wide variety of data
formats and modalities. Many existing specialized solutions are hyper-specialized and support only
specific modalities (e.g., RecordIO datasets can only store images with associated floating-point
labels), while others are slightly more flexible (e.g., TFRecord and WebDataset). In FFCV, we opt
for maximal flexibility, and use an abstract “Field” class that enables users to store arbitrary data
modalities (with built in support for vision, text, tabular, and more), and even easily extend FFCV’s
capabilities by writing data-specific customized encoders and decoders.
Searchability/Indexability. A good data format should also natively support fast access to only a
particular subset of the dataset, whether for the purpose of inspecting a given example, or training
on a particular subset of the training set. Specialized data formats that only support sequential
4

Figure 3: Structure of a .beton file used to store a simple image classification dataset.
reads, however (e.g., TFRecord, WebDataset) are inherently unable to support such a feature. FFCV
datasets contain a data table that hold metadata (including indices) as well as pointers to any given
sample, allowing one to easily filter and retrieve samples based on any predicate.
File structure. FFCV datasets are optimized for machine learning training and offer great performance regardless of the underlying storage method (RAM, HDDs, SSDs or network). Each
file consists of four sections. The Header contains general information about the dataset like the
number of samples and the fields. The Data Table is a small DataFrame-like data structure containing
metadata (small fixed-width information) about a given sample, such as e.g. the image resolution
in the image domain, or audio sample duration in the audio domain. The Heap Storage section
contains pages (of default size 8MB) that store either variable size information or data that is too
large for the Data Table, such as binary representations of images/audio examples. Finally, the
Allocation Table at the end of the file has bookkeeping data about allocated regions in Heap Storage.
We show what a .beton file would look like for a basic image classification task in Figure 3.

3.2

Challenge #2: Efficient Data Reading

We now describe how FFCV achieves high read performance across a variety of compute environments with the FFCV file format, ranging from those featuring local SSDs (with high IOPS and low
latency) to large spinning disks (which suffer under random, nonsequential accesses, such as when
reading many discrete image files) to networked filesystems. FFCV offers built-in read strategies
optimized for high throughput across all these different scenarios.
Operating system caching. For systems that can fit the dataset in random-access memory (RAM),
FFCV can take advantage of OS-level caching. This ensures that every data read after the first
one will be from RAM rather than disk, resulting in high throughput. Beyond just simplicity,
OS-level caching also allows for multiple models training in parallel on the same dataset (i.e., when
hyperparameter searching) to share the same cache without any additional memory overhead.
Process cache. On the other hand, if the dataset is larger than the main memory, an effective
caching scheme will have to optimally cache, discard, and reload data at each epoch. In OS-level
caching—which other data loading schemes use by default—the random access patterns induced
by SGD in machine learning cause suboptimal caching behavior. FFCV circumvents this issue
5

Figure 4: Illustration of the procedure followed by FFCV to generate the code of a complex image
processing pipeline. Transforms are categorized together based on whether they can be JIT-ed
(dashed, FFCV native or numpy based user-defined augmentations) or not (solid, Pytorch ones and
others). Groups (stages) are formed based on these categories (1-4) by gluing each operation using
meta-programming. Finally, the stages are compiled to machine code using Numba/LLVM.
through optimized, process-level caching. By leveraging our knowledge of the sample order in
data loading (since we can generate this order at the beginning of the epoch), FFCV can preload
data much earlier than the OS can.
Quasi-random sampling. For cases where disk reads are particularly expensive (e.g., when
reading from a network drive and having insufficient RAM to cache the dataset), FFCV offers a
quasi-random loading strategy that can combine with the process cache strategy above to minimize
the stress on the underlying storage. Rather than reading examples in a uniformly random order,
the quasi-random strategy (a) allocates a buffer large enough to fit batch size pages of the dataset;
(b) samples a permutation of all the pages of the dataset; then (c) generates a batch only from
samples in the buffer.
WebDataset’s shuffling procedure is similar to FFCV’s quasi-random loading strategy with
two crucial differences: (1) pages in FFCV are much smaller than WebDataset shards, leading
to significantly better randomness; 1 (2) quasi-random loading in FFCV has a constant memory
footprint, while WebDataset’s footprint scales linearly in the number of workers. For further
comparison, see Appendix A.

3.3

Challenge #3: Fast Data Processing

So far, we have outlined how FFCV improves both data storage and reading—we now turn to the data
processing stage of the machine learning pipeline. In ML research, the data augmentation/processing
pipeline requires both efficiency (to avoid bottlenecking the entire training process) and flexibility
(to accommodate, e.g., researchers devising their own augmentations or pre-processing techniques).
FFCV tries to strike a balance between these two objectives through a just-in-time (JIT) compiled
data processing pipeline. Specifically, for a small cost paid at the start of training, FFCV analyzes
the user-provided (Python) data processing pipeline, and automatically compiles it to optimized
machine code via the following steps:
1While one can manually make WebDataset shards smaller, this incurs a significant filesystem load.

6

Categorization. Our main tool for compiling Python to machine code is the Numba library
[LPS15], which is by default able to compile a large (but not complete) subset of the Python
language into machine code.2 We thus first categorize each element of the data pipeline based on
whether it can be automatically compiled by Numba. (Note that all the pipeline elements that
ship with FFCV are Numba-compilable, so this step is primarily to enable users to write their own
FFCV compatible non-compilable transformations, as in cases where it is too difficult to write a
compilable verfsion of a transformation).
Grouping. After categorizing each transformation in the data pipeline, we group together all
consecutive transforms of each category into groups called “stages” (see Figure 4). This will allow
us to compile several separate pipeline elements into a single executable block of machine code.
Code generation. Finally, using meta-programming, we generate the code necessary to fuse each
stage into a single function. Some of the stages will be then passed to Numba to be converted to
machine code, the others remain unmodified and run natively in Python (albeit at much lower
speed than their compiled counterparts).
Memory pre-allocation. A core tenet of FFCV is to avoid unnecessary memory allocation. Thus,
every operation in the pipeline declares memory requirements in advance, and memory allocation
is performed once at the start of an epoch. To let workers prepare the data while training happens,
and to absorb potential slow downs in data preparation, FFCV relies on a circular buffer (illustrated
in Figure 7 in Appendix D).

3.4

Challenge #4: Circumventing Data Transfer Costs

Since compiled machine code is not under the supervision of the Python interpreter, FFCV can
escape the constraints of Python’s global interpreter lock (GIL) and can rely on threads instead of
sub-processes like most libraries (the GIL typically only allows a single thread to use the Python
interpreter at once, generally making multi-threading infeasible).
Threads yields two important advantages for FFCV. First, threads can collaborate directly by
reading/writing memory instead of using (expensive) communication primitives. FFCV workers
can therefore work together on same batch instead of having to work on their own, improving
on latency and saving large quantities of memory (i.e., FFCV’s memory usage is typically constant
instead of scaling with the number of workers). Second, since they share the same CUDA context,
all data preparation operations running on GPUs (e.g., data copying, augmentations) can be run
asynchronously and—critically—in parallel with respect to the training loop, reducing the length
of the critical path. See Appendix B for more details.

4

Case Studies

In this section, we showcase FFCV’s versatility by illustrating how it can dramatically accelerate
model training in three common practical settings. While one can use FFCV with any task or
modality, we center our first three use cases around the ImageNet ILSVRC-2012 image classification
task, which comprises 1.3 million labeled training images corresponding to 1,000 different classes.
ImageNet is a standard dataset in both image classification [RDS+15; HZR+15; KSH12] and model
training speed benchmarks [MCC+20; CNK+17; CKN+19]; indeed, searching “ImageNet PyTorch”
on GitHub returns hundreds of thousands of repositories.
2We motivate this choice and compare with other compilation systems in Appendix C.

7

We show that FFCV enables dramatic speedups over typical setups3 in the following cases:
• Single-model training: We first use FFCV to train a ResNet-50 on ImageNet to 75% accuracy in
20 minutes on a single node.
• Multi-model training: We then consider the setting where a researcher wants to train several
small models in parallel (e.g., to obtain confidence intervals or perform hyperparameter search).
We show that FFCV can train 8 ResNet-18s at the same time (one per GPU) without incurring
any additional overhead over single-GPU training. This demonstrates that FFCV enables both (a)
efficient training of high throughput models and (b) low-overhead concurrent training.
• Low-memory training: Finally, we consider the (practically common) setting in which the
dataset does not fit into machine memory (RAM). Via process-level caching and a quasi-random
sampling scheme (exposed to users via just two lines of code), FFCV accelerates training even
when reading data from slow disks (and even from networked file systems) with minimal
performance overhead.
We additionally demonstrate FFCV’s drop-in applicability beyond computer vision tasks:
• GPU-enabled sparse regression: In just a few lines of code, FFCV can considerably speed up an
iterative SAGA [DBL14] solver (similar to that of Wong et al. [WSM21]) by simply replacing a
default PyTorch data loader (loading from a memory-mapped file) with an FFCV loader.

4.1

Training a single model

We first study the simplest use case of FFCV: training a single model on ImageNet as fast as possible.
By combining the data loading speed of FFCV with known ImageNet training optimizations, we
are able to establish a competitive speed/accuracy tradeoff for the benchmark task of training a
ResNet-50 on ImageNet (Figure 1).
Fast training. We begin with an overview of the training algorithm itself—a long line of work
has explored various modifications to standard training that have been shown to improve speed
and/or accuracy, of which we use Blurpool [Zha19]; NoWD-BN [JSH+18]; linear learning rate
annealing [LYR20]; test-time augmentation and resizing [TVD+19]; and progressive resizing (i.e.,
we start training at 160px resolution and then increase to 192px 75% of the way through training).
Fast data loading. With FFCV we JPEG-compress 50% of the dataset, compromising between
compute (i.e. faster image processing, as 50% of the images come pre-decoded) and available
memory. Doing so allows our system to:
(a) reap the full benefits of progressive resizing. Even at smaller resolutions like 160px in which
the GPU has much higher throughput, we can still fully saturate the GPU as we are neither
data reading nor processing bottlenecked;
(b) outsource augmentations to the CPU. Since the dataset is cached (largely decoded) in memory,
we can use CPU cycles for augmentations that would otherwise go to decoding.
3 In this report, we compare FFCV to the standard PyTorch dataloader rather than specialized solutions like NVIDIA

DALI (which tend to be less versatile and significantly harder to use). Our goal is to illustrate how a simple drop-in
replacement enables greatly accelerated training in a variety of settings. For more benchmarks, see docs.ffcv.io.

8

256
128
64

(a) Single-model (1x RN-50 on
8xA100)

1024
512

FFCV
ImageFolder

256
128
64

(b) Multi-model (8x RN-18 on
1xA100 each)

Training Time (sec/epoch)

512

FFCV
ImageFolder

Training Time (sec/epoch)

Training Time (sec/epoch)

1024

1024

FFCV
ImageFolder

512
256
128
64

(c) Low-memory (1x ResNet-18
via Network File System)

Figure 5: A comparison between the time to train one epoch on ImageNet using FFCV and PyTorch’s
ImageFolder.
Evaluation.

We compare our FFCV-enabled optimized ImageNet example to these baselines:

• PyTorch ImageNet example: As a naive baseline, we take the PyTorch ImageNet example
code, which is both unoptimized and slow to load data—we use the reported accuracy from
torchvision and multiply the per-epoch time (from Figure 5a by 90 to obtain an optimistic
estimate of total training time. We modify the code to use half precision. This loader is far and
away the most popular loader seen in open source/research implementations, and is used by
the most popular open-source ImageNet training libraries [Wig19; Fal19; Dev16].
• MosaicML: Finally, we consider models trained with the MosaicML Composer [Tea21] training
system (as of Februrary 20224 ) as a baseline. MosaicML Composer supports many kinds
of training options, including MixUp [ZCD+17], specialized optimizers [FKM+21], squeezeexcitation blocks [HSS18], and more. We compare against the pareto optimal MosaicML Composertrained models in terms of training speed and accuracy (Team [Tea21] study models trained
across combinatorial choices of training techniques).
We run all implementations on an AWS EC2 p4d.24xlarge machine. We report our results in
Figure 1; our system obtains the best accuracy vs. speed trade-off across all baselines. In particular,
to obtain 75% accuracy we require only 20 minutes, much faster than any of the tested baselines.

4.2

Training multiple models

Another common paradigm in machine learning is training multiple models simultaneously. For
example, we may want to perform a grid search for optimal parameters, or rerun a model with the
same training parameters to obtain confidence intervals on results. In what follows, we show that
FFCV allows for much faster parallel training than existing methods: FFCV has automatic support for
OS-level caching and is high throughput enough to support even eight ResNet-18 models training
simultaneously (ResNet-18 models have nearly three times the throughput of ResNet-50 models
since they are smaller).
Evaluation. Using the same AWS EC2 p4d.24xlarge machine as above, we run eight concurrent
training routines on ResNet-18 models using FFCV and compare with the PyTorch ImageNet
example baseline (originally described in Section 4.1).
Each training routine has access to one eighth of the available vCPUs (12) and one A100. For
FFCV, we use image datasets that have been originally scaled to 350px, and we perform no JPEG
4 Since the release of FFCV, the MosaicML Composer has replaced its dataloader with FFCV—for the sake of comparison,
we thus use the pre-FFCV version of the Composer.

9

samples / sec

16000
14000

Loader

PyTorch
FFCV

12000
10000
8000

500

1000

1500

2000

2500

Batch size

3000

3500

4000

Figure 6: FFCV provides significant speedups over the default PyTorch data loader, even for nonvision modalities. Here, we replace only the dataloader of a sparse regression solver with FFCV.
This simple change makes the solver 1.6 times faster.
compression, storing only image pixel values. Our throughput results can be found in Figure 5b;
we find that FFCV has greater throughput than PyTorch’s default ImageFolder loader, despite not
requiring any specialized hardware for decoding.

4.3

Low-memory training

In the previous two examples, we operated in a setting where the machine being used for training
has sufficient memory (RAM) to cache the entire ImageNet dataset (in particular, our 50% compressed version of ImageNet is 339GB). In many scenarios, however, we do not have sufficient
RAM to cache even a fully JPEG-compressed dataset, and are thus forced to load images directly
from the filesystem. Normally, this process incurs additional significant training cost, especially in
settings where the filesystem is mounted on a networked drive (or any other slow disk).
Here, we show that with minimal changes to existing code, FFCV enables fast training even
in such resource-constrained setups. By changing only two lines of code, we can enable processlevel caching and quasi-random loading. These optimizations together ensure that read-constrained,
memory-constrained systems operate at as high a throughput as possible; see Section 3 for details.
Evaluation. Just as in the last section, we compare to the default PyTorch loader. The results,
shown in Figure 5c, illustrate that FFCV indeed enables faster training in memory-limited settings.

4.4

Beyond computer vision

Finally, we show the applicability of FFCV beyond just computer vision workloads. Specifically,
we consider a large-scale sparse linear regression problem with n = 100, 000 training points and
dimensionality d = 50, 000. We use an optimized iterative SAGA-based optimizer [WSM21] for
solving sparse linear regression problems. In Figure 6, we compare the unmodified code (which
loads data using the standard PyTorch data loader, reading from a memory-mapped file) with a
drop-in FFCV replacement. The results indicate that even outside typical computer vision setups,
FFCV is an effective drop-in replacement for default data loaders.

10

5

Related Work

In this section, we discuss some prior work in the field of accelerating machine learning training.
Data pipelines in ML. Mohan et al. [MPR+21] find that DNN training time is dominated by data
loading in various settings. DALI [NVI18] uses custom input data pre-processing pipelines, with
the option of off-loading some work to the GPU. [AMB19] introduce AIStore, a storage system,
and WebDataset, a storage format based on POSIX tar, to enable high performance I/O for large
scale deep learning. Murray et al. [MSK+21] analyze millions of jobs on Google cloud and find
that they spend a significant fraction of time in the input pipeline; they find that optimizing input
pipeline performance is critical to end-to-end training time. Their framework tf.data allows
users to build and execute efficient input pipelines, assisting with parallelism, caching, and static
optimizations. Kakaraparthy et al. [KVP+19] find that ML experiments with concurrent jobs (such
as grid search) benefit from unifying data loading across jobs. Alternatively, kornia [RMP+20]
implements standard image processing functions for GPUs.
Speedups from other sources. While our work and those cited above focus on removing the
data bottlenecks in current ML workloads, large speed improvements also come from other
sources, including hardware and algorithmic improvements, which allow models to achieve
similar accuracies with less training. These include better architectures (e.g., ResNet [HZR+15]),
optimizations (e.g., batch normalization [IS15], cyclic LR [Smi17]), data augmentation (e.g., MixUp
[ZCD+17]), among others.
Fast training on ImageNet. Our evaluation focuses on fast training on ImageNet, a standard in
model training speed benchmarks [MCC+20; CNK+17; CKN+19]. Many prior works [GDG+17;
JSH+18; YZH+17; ASF17] use distributed training with extremely large batch sizes to reduce training
time. Beyond the increase in engineering complexity arising from distributed training, training
models with large batch sizes comes with its own challenges, for instance, proper tuning of the
learning rate [DZ19; YZH+17]. Moreover, extreme resource usage, including large communication
overheads necessitated by distributed training [CKN+19], reduce their usability.

6

Conclusion

In this work, we present FFCV, an optimized framework for eliminating data bottlenecks in machine
learning model training routines. We use FFCV to substantially improve speed/accuracy tradeoff
for the ImageNet dataset, and demonstrate (through a series of case studies) the potential for FFCV
to speed up almost any ML workload. The main limitation of our work is that it does not address
non-data related bottlenecks in training, and might thus yield less significant (but still non-zero)
improvements in settings where data is fast to load and process (e.g., natural language processing)
or where models are very large and dominate training time (e.g., large pre-trained vision models).

7

Acknowledgements

Work supported in part by the NSF grants CNS-1815221 and DMS-2134108, and Open Philanthropy.
This material is based upon work supported by the Defense Advanced Research Projects Agency
(DARPA) under Contract No. HR001120C0015.
11

References
[AMB19]

Alex Aizman, Gavin Maltby, and Thomas Breuel. “High performance I/O for large
scale deep learning”. In: 2019 IEEE International Conference on Big Data (Big Data). IEEE.
2019, pp. 5965–5967.

[ASF17]

Takuya Akiba, Shuji Suzuki, and Keisuke Fukuda. “Extremely large minibatch sgd:
Training resnet-50 on imagenet in 15 minutes”. In: arXiv preprint arXiv:1711.04325
(2017).

[CKN+19]

Cody Coleman et al. “Analysis of DAWNBench, a Time-to-Accuracy Machine Learning
Performance Benchmark”. In: Operating Systems Review (2019).

[CNK+17]

Cody Coleman et al. “Dawnbench: An end-to-end deep learning benchmark and
competition”. In: NeurIPS ML Systems Workshop. 2017.

[DBL14]

Aaron Defazio, Francis Bach, and Simon Lacoste-Julien. “SAGA: A fast incremental
gradient method with support for non-strongly convex composite objectives”. In:
Advances in neural information processing systems (NeurIPS). 2014.

[DDS+09]

Jia Deng et al. “Imagenet: A large-scale hierarchical image database”. In: Computer
Vision and Pattern Recognition (CVPR). 2009.

[Dev16]

PyTorch Developers. PyTorch ImageNet Example. https : / / github . com / pytorch /
examples/tree/master/imagenet. 2016.

[DZ19]

Tim Dettmers and Luke Zettlemoyer. “Sparse networks from scratch: Faster training
without losing performance”. In: arXiv preprint arXiv:1907.04840 (2019).

[Fal19]

William Falcon et al. “PyTorch Lightning”. In: GitHub. Note: https://github.com/PyTorchLightning/pytorchlightning 3 (2019).

[FKM+21]

Pierre Foret et al. “Sharpness-Aware Minimization for Efficiently Improving Generalization”. In: International Conference on Learning Representations (ICLR). 2021.

[GDG+17]

Priya Goyal et al. “Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour”. In:
arXiv preprint 1706.02677 (2017).

[HSS18]

Jie Hu, Li Shen, and Gang Sun. “Squeeze-and-excitation networks”. In: Proceedings of
the IEEE conference on computer vision and pattern recognition. 2018.

[HZR+15]

Kaiming He et al. Deep Residual Learning for Image Recognition. 2015.

[IS15]

Sergey Ioffe and Christian Szegedy. “Batch Normalization: Accelerating Deep Network
Training by Reducing Internal Covariate Shift”. In: International Conference on Machine
Learning (ICML). 2015.

[JSH+18]

Xianyan Jia et al. “Highly scalable deep learning training system with mixed-precision:
Training imagenet in four minutes”. In: arXiv preprint arXiv:1807.11205 (2018).

[KSH12]

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. “Imagenet Classification with
Deep Convolutional Neural Networks”. In: Advances in Neural Information Processing
Systems (NeurIPS). 2012.

[KVP+19]

Aarati Kakaraparthy et al. “The case for unifying data loading in machine learning
clusters”. In: 11th {USENIX} Workshop on Hot Topics in Cloud Computing (HotCloud 19).
2019.

12

[LPS15]

Siu Kwan Lam, Antoine Pitrou, and Stanley Seibert. “Numba: A llvm-based python jit
compiler”. In: Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in
HPC. 2015, pp. 1–6.

[LYR20]

Mengtian Li, Ersin Yumer, and Deva Ramanan. “Budgeted training: Rethinking deep
neural network training under resource constraints”. In: International Conference on
Learning Representations. 2020.

[MCC+20]

Peter Mattson et al. “MLPerf Training Benchmark”. In: MLSys. 2020.

[MPR+21]

Jayashree Mohan et al. “Analyzing and Mitigating Data Stalls in DNN Training”. In:
Proc. VLDB Endow. 14.5 (Jan. 2021), pp. 771–784. ISSN: 2150-8097. DOI: 10 . 14778 /
3446095.3446100. URL: https://doi.org/10.14778/3446095.3446100.

[MSK+21]

Derek G Murray et al. “tf. data: A Machine Learning Data Processing Framework”. In:
arXiv preprint arXiv:2101.12127 (2021).

[NVI18]

NVIDIA. DALI: NVIDIA Data Loading Library. 2018. URL: https://developer.nvidia.
com/dali.

[PGM+19]

Adam Paszke et al. “PyTorch: An Imperative Style, High-Performance Deep Learning
Library”. In: Advances in Neural Information Processing Systems 32. Ed. by H. Wallach
et al. Curran Associates, Inc., 2019, pp. 8024–8035. URL: http://papers.neurips.cc/
paper/9015-pytorch-an-imperative-style-high-performance-deep-learninglibrary.pdf.

[RDS+15]

Olga Russakovsky et al. “ImageNet Large Scale Visual Recognition Challenge”. In:
International Journal of Computer Vision (IJCV). 2015.

[RMP+20]

Edgar Riba et al. “Kornia: an open source differentiable computer vision library for
pytorch”. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer
Vision. 2020, pp. 3674–3683.

[Smi17]

L. N. Smith. “Cyclical Learning Rates for Training Neural Networks”. In: Winter
Conference on Applications of Computer Vision. 2017.

[Tea21]

The Mosaic ML Team. composer. https://github.com/mosaicml/composer/. 2021.

[TVD+19]

Hugo Touvron et al. “Fixing the train-test resolution discrepancy”. In: arXiv preprint
arXiv:1906.06423 (2019).

[Wig19]

Ross Wightman. PyTorch Image Models. https://github.com/rwightman/pytorchimage-models. 2019. DOI: 10.5281/zenodo.4414861.

[WSM21]

Eric Wong, Shibani Santurkar, and Aleksander Madry. “Leveraging Sparse Linear Layers for Debuggable Deep Networks”. In: International Conference on Machine Learning
(ICML). 2021.

[YZH+17]

Yang You et al. “100-epoch imagenet training with alexnet in 24 minutes”. In: arXiv
preprint arXiv:1709.05011 (2017).

[ZCD+17]

Hongyi Zhang et al. “mixup: Beyond empirical risk minimization”. In: arXiv preprint
arXiv:1710.09412 (2017).

[Zha19]

Richard Zhang. “Making convolutional networks shift-invariant again”. In: International conference on machine learning. PMLR. 2019, pp. 7324–7334.

13

A

Quasi-Randomness vs. Webdataset Sharding

In Section 3.2, we mentioned that, while FFCV’s quasi-random loading strategy is similar to WebDataset’s sharding, the two strategies differ in key aspects. Here we discuss in more detail why this
is the case.
In systems like Webdataset, sharding is done to reduce the number of files and relieve some
strain on the file system. However, each file system has a block size after which random reads
become sequential reads. In almost all system, this lies below 2MB. So, if Webdataset was to create
shards of 2MB, it would create a dataset with way too many files that it overwhelms the file system.
On the other hand, FFCV’s .beton format (by design) decouples this phenomenon by sharding
internally a single large file. This means that our dataset is a single file the is structured to imitate
sharding, as shown in Figure 3. This enables us to increase the quality of randomness as our
shards are typically orders of magnitude smaller than WebDataset’s. At the same time, we avoid
overwhelming the filesystem as long as we pick the minimum shard size that will satisfy the file
system while retaining a single file.

B

Why does FFCV Use Multithreading Instead of Multiprocessing?

In this section, we discuss in more detail (than Section 3.4) why FFCV relies to threads instead of
sub-processes.
While many libraries, such as PyTorch, use multiprocessing during data loading to avoid the
GIL, this comes with a large performance drop. In such strategies, all workers have to report back
to the main process (running on a single thread). This often leads to lower performance with very
large worker pools than with smaller as the main thread becomes overwhelmed. To avoid this, we
resorted in FFCV to multi-threading instead, thus enabling each thread to write an individual sample
directly in place in RAM without any inter-process communication. This also enables multiple
threads to work simultaneously on the same batch, and avoids situations like when PyTorch’s main
thread could be sitting idle waiting for any of its workers to have their batches ready, although if
these workers could accumlate their work (which is what we do in FFCV), the main thread would
have not have been idle.

C

Comparison of JIT Compilers Alternatives

We motivate our decision to use Numba to compile FFCV’s pipelines by elaborating on the pros
and cons of potential alternatives:
torch.script: This JIT compiler, included as part of Pytorch was designed to optimize inference
speed of deep neural networks and ease the transition between research code and deployment
environments. In this context, it makes sense for it to only support the most basic python features
and the Pytorch library itself. Relying on this solution for FFCV would have mean giving up on
data augmentations written in numpy and interoperability with other deep learning frameworks.
While torch.scrip can generate very fast code for some ML oriented operations (e.g matrix
multiplications), it lacks some optimizations (for loops) and does not release the GIL which makes
would have made it particularly unsuited for FFCV.
It is still important to note that it still possible to leverage torch.script within FFCV: users are
allowed to introduce in their pipeline functions compiled with it as long as they operate on the

14

GPU. Indeed, FFCV since does not compile those, it does not interacting with Numba and could
even bring additional performance improvements in some scenarios.
TVM: TVM5 shares many properties with Numba. They both are capable to generate LLVM IR,
and can leverage the same optimization passes. Their performance should in theory be very similar.
TVM has other advantages like being able to target other environments, this would not be useful
for FFCV. The main differentiating factor remaining is the API. The familiar numpy augmented
python seemed to be the most approachable for potential users. On the other end, TVM which was
designed to represent and optimize complex neural network architectures would have been much
more cumbersome to work with in this scenario.

D

Omitted Figures

Figure 7: Illustration of FFCV’s circular buffer. The producer (data processing pipeline) works on
an entry of the buffer while the consumer (user’s code) uses a previously filled one. When done,
they moving clockwise and pause when they encounter each other.

5 https://tvm.apache.org/

15

