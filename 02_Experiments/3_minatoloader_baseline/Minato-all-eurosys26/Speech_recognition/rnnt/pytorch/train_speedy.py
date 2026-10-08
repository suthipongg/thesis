# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch.multiprocessing import Process, Queue, set_start_method, Event
import random 
import argparse
import copy
import os
import csv
import random
import time
from common.data.Async import AsynchronousLoader
import torch
import numpy as np
import torch.distributed as dist
from apex import amp
from apex.optimizers import FusedLAMB
from torch.utils.data.distributed import DistributedSampler

from apex.parallel import DistributedDataParallel
from common.data.speedy_dataset import AudioDataset
from common import helpers
from common.data.speedy_dataset import get_data_loader
from common.data.text import Tokenizer
from common.data import features
from common.helpers import (Checkpointer, greedy_wer, num_weights, print_once,
                            process_evaluation_epoch)
from common.optimizers import lr_policy
from common.tb_dllogger import flush_log, init_log, log
from rnnt import config
from rnnt.decoder import RNNTGreedyDecoder
from rnnt.loss import RNNTLoss
from rnnt.model import RNNT

from mlperf import logging

import time
import random
from torch.multiprocessing import Process, Queue
from queue import Empty




def preprocess_audio1(waveform, n_fft=400, hop_length=160, n_mels=240):
   
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Ensure batch dimension: (1, audio_length)

    batch_size, audio_length = waveform.shape
    print(waveform.shape)

    stft_output = torch.stft(
        waveform, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        return_complex=True
    )  # Shape: (batch_size, freq_bins, time_steps)

    magnitude_spectrogram = stft_output.abs()  # Remove phase

    if magnitude_spectrogram.dim() == 2:  # If shape is (freq_bins, time_steps), add batch dim
        magnitude_spectrogram = magnitude_spectrogram.unsqueeze(0)

    spectrogram_resized = torch.nn.functional.interpolate(
        magnitude_spectrogram.unsqueeze(1),  # Add channel dim (batch, 1, freq_bins, time_steps)
        size=(n_mels, magnitude_spectrogram.shape[-1]), 
        mode="bilinear",
        align_corners=False
    ).squeeze(1)  # Remove channel dim → (batch, 240, time_steps)

    spectrogram_final = spectrogram_resized.permute(0, 2, 1)  # (batch_size, time_steps, 240)

    return spectrogram_final.squeeze(0)  # Remove batch dim if needed





def collate_fn(batch):
    # Assuming each sample in the batch is a tuple (audio_tensor, audio_len, transcript_tensor, transcript_len)
    bs = len(batch)
    print("Batch size:", bs)

    # Debugging the structure of the batch
    for i, sample in enumerate(batch):
        if not isinstance(sample, tuple) or len(sample) != 4:
            print(f"Warning: Batch sample {i} is not in the expected tuple format!")
            continue
        audio_tensor, audio_len, transcript_tensor, transcript_len = sample
        
        # Check if audio_tensor and transcript_tensor are Tensors
        if not isinstance(audio_tensor, torch.Tensor) or not isinstance(transcript_tensor, torch.Tensor):
            print(f"Warning: Sample {i} has invalid tensor format!")
            continue

        print(f"Sample {i}: Audio tensor shape = {audio_tensor.shape}, Transcript tensor shape = {transcript_tensor.shape}")

    # Lambda functions to calculate max lengths for audio and transcript tensors
    max_audio_len = lambda l: max(el[0].size(0) for el in l if isinstance(el[0], torch.Tensor))  # Max length of audio
    max_transcript_len = lambda l: max(el[2].size(0) for el in l if isinstance(el[2], torch.Tensor))  # Max length of transcript

    # Create tensors to hold the audio and transcript data
    audio = torch.zeros(bs, max_audio_len(batch))  # Tensor for audio data
    audio_lens = torch.zeros(bs, dtype=torch.int32)  # Tensor for audio lengths
    transcript = torch.zeros(bs, max_transcript_len(batch))  # Tensor for transcripts
    transcript_lens = torch.zeros(bs, dtype=torch.int32)  # Tensor for transcript lengths

    for i, sample in enumerate(batch):
        audio_tensor, audio_len, transcript_tensor, transcript_len = sample

        # Populate the audio tensor with the correct sample data
        audio[i].narrow(0, 0, audio_tensor.size(0)).copy_(audio_tensor)
        audio_lens[i] = audio_len

        # Populate the transcript tensor with the correct sample data
        transcript[i].narrow(0, 0, transcript_tensor.size(0)).copy_(transcript_tensor)
        transcript_lens[i] = transcript_len

        # Debugging print statements
        print("Speedyloader: audio=", audio.shape)
    
    # If there's a preprocessing step for the audio data, apply it
    audio = preprocess_audio1(audio)

    # Debugging print statements for shapes after processing
    print("Audio shape after preprocessing:", audio.shape)
    print("Transcript shape:", transcript.shape)
    
    # Return the processed tensors
    return audio, audio_lens, transcript, transcript_lens


class DataProducer(Process):
    def __init__(self, queue, dataset, indices, rank):
        super().__init__()
        self.queue = queue
        self.dataset = dataset
        self.indices = indices
        self.rank = rank

    def run(self):
        for idx in self.indices:
            self.process_sample(idx)

    def process_sample(self, idx):
        """Handles loading and queuing of a single sample."""
        try:
            batch = self.dataset[idx]  # Fetch sample from dataset
            
            if batch == 'heavy':
                print(f"Skipping heavy sample at index {idx}. It will be fetched later from queue timeout.")
                return

            # Wait until the queue size is below the threshold before adding
            while self.queue.qsize() > 97:
                time.sleep(1)  # Small sleep to allow processing

            # Enqueue the batch
            print("add sample to the queue", self.queue.qsize())
            self.queue.put((batch, self.rank))  # Blocks if queue is full

        except Exception as e:
            print(f"Error processing index {idx}: {e}")  # Log error

        












def parse_args():
    parser = argparse.ArgumentParser(description='RNN-T Training Reference')

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', default=100, type=int,
                          help='Number of epochs for the entire training')
    training.add_argument("--warmup_epochs", default=6, type=int,
                          help='Initial epochs of increasing learning rate')
    training.add_argument("--hold_epochs", default=40, type=int,
                          help='Constant max learning rate epochs after warmup')
    training.add_argument('--epochs_this_job', default=0, type=int,
                          help=('Run for a number of epochs with no effect on the lr schedule.'
                                'Useful for re-starting the training.'))
    training.add_argument('--cudnn_benchmark', action='store_true', default=True,
                          help='Enable cudnn benchmark')
    training.add_argument('--amp', '--fp16', action='store_true', default=False,
                          help='Use mixed precision training')
    training.add_argument('--seed', default=None, type=int, help='Random seed')
    training.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                          help='GPU id used for distributed training')
    training.add_argument('--target', default=0.058, type=float, help='Target WER accuracy')
    training.add_argument('--weights_init_scale', default=0.5, type=float, help='If set, overwrites value in config.')
    training.add_argument('--hidden_hidden_bias_scale', type=float, help='If set, overwrites value in config.')

    optim = parser.add_argument_group('optimization setup')
    optim.add_argument('--batch_size', default=128, type=int,
                       help='Effective batch size per GPU (might require grad accumulation')
    optim.add_argument('--val_batch_size', default=2, type=int,
                       help='Evalution time batch size')
    optim.add_argument('--lr', default=4e-3, type=float,
                       help='Peak learning rate')
    optim.add_argument("--min_lr", default=1e-5, type=float,
                       help='minimum learning rate')
    optim.add_argument("--lr_exp_gamma", default=0.935, type=float,
                       help='gamma factor for exponential lr scheduler')
    optim.add_argument('--weight_decay', default=1e-3, type=float,
                       help='Weight decay for the optimizer')
    optim.add_argument('--grad_accumulation_steps', default=8, type=int,
                       help='Number of accumulation steps')
    optim.add_argument('--log_norm', action='store_true',
                       help='If enabled, gradient norms will be logged')
    optim.add_argument('--clip_norm', default=1, type=float,
                       help='If provided, gradients will be clipped above this norm')
    optim.add_argument('--beta1', default=0.9, type=float, help='Beta 1 for optimizer')
    optim.add_argument('--beta2', default=0.999, type=float, help='Beta 2 for optimizer')
    optim.add_argument('--ema', type=float, default=0.999,
                       help='Discount factor for exp averaging of model weights')

    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--dali_device', type=str, choices=['cpu', 'gpu'],
                    default='cpu', help='Use DALI pipeline for fast data processing')
    io.add_argument('--resume', action='store_true',
                    help='Try to resume from last saved checkpoint.')
    io.add_argument('--ckpt', default=None, type=str,
                    help='Path to a checkpoint for resuming training')
    io.add_argument('--save_at_the_end', action='store_true',
                    help='Saves model checkpoint at the end of training')
    io.add_argument('--save_frequency', default=None, type=int,
                    help='Checkpoint saving frequency in epochs')
    io.add_argument('--keep_milestones', default=[], type=int, nargs='+',
                    help='Milestone checkpoints to keep from removing')
    io.add_argument('--save_best_from', default=200, type=int,
                    help='Epoch on which to begin tracking best checkpoint (dev WER)')
    io.add_argument('--val_frequency', default=1, type=int,
                    help='Number of epochs between evaluations on dev set')
    io.add_argument('--log_frequency', default=25, type=int,
                    help='Number of steps between printing training stats')
    io.add_argument('--prediction_frequency', default=None, type=int,
                    help='Number of steps between printing sample decodings')
    io.add_argument('--model_config', default='configs/baseline_v3-1023sp.yaml',
                    type=str, required=True,
                    help='Path of the model configuration file')
    io.add_argument('--num_buckets', type=int, default=6,
                    help='If provided, samples will be grouped by audio duration, '
                         'to this number of backets, for each bucket, '
                         'random samples are batched, and finally '
                         'all batches are randomly shuffled')
    io.add_argument('--train_manifests', type=str, required=True, nargs='+',
                    help='Paths of the training dataset manifest file')
    io.add_argument('--val_manifests', type=str, required=True, nargs='+',
                    help='Paths of the evaluation datasets manifest files')
    io.add_argument('--max_duration', type=float,
                    help='Discard samples longer than max_duration')
    io.add_argument('--dataset_dir', required=True, type=str,
                    help='Root dir of dataset')
    io.add_argument('--output_dir', type=str, required=True,
                    help='Directory for logs and checkpoints')
    io.add_argument('--log_file', type=str, default=None,
                    help='Path to save the training logfile.')
    io.add_argument('--max_symbol_per_sample', type=int, default=None,
                    help='maximum number of symbols per sample can have during eval')
    return parser.parse_args()


# Function for generating batches and adding them to a queue
import time

# Function for generating batches and adding them to a queue
def batch_constructor(queue, queue_timeout, batch_queue, batch_size, stop_event):
    try:    
        while not stop_event.is_set():
            batch_data = []
            while len(batch_data) < batch_size:
                print(f"Queue size: {queue.qsize()}, Timeout queue size: {queue_timeout.qsize()}")


                # Always prioritize queue first, but make sure timeout queue is also used
                if not queue.empty():
                    item, rank = queue.get()
                    print('item from queue')
                elif not queue_timeout.empty(): # If queue is empty, dequeue from timeout queue
                    item, rank = queue_timeout.get()
                    print('item from queue_timeout')

               
                else:
                    time.sleep(0.01)
                    continue
                
                if item is None:
                    print("Received sentinel. Exiting.")
                    raise StopIteration

                

                batch_data.append(item)
            while batch_queue.qsize() >=99:
                print("batch_qeueue is almost full")
                time.sleep(2)

            batch_queue.put(batch_data)
            print('put in batch_queue', batch_queue.qsize())


    except Exception as e:
        print(f"Exception in batch_constructor: {e}")


    finally:
        batch_queue.put(None) 


def apply_ema(model, ema_model, decay):
    if not decay:
        return

    sd = getattr(model, 'module', model).state_dict()
    for k, v in ema_model.state_dict().items():
        v.copy_(decay * v + (1 - decay) * sd[k])




def calculate_tensor_size(tensor):
    """Calculate the size of a PyTorch tensor in MB."""
    if tensor is None:
        return 0.0  # Handle cases where tensor might be None
    num_elements = tensor.numel()  # Total number of elements
    bytes_per_element = tensor.element_size()  # Size of each element in bytes
    size_in_bytes = num_elements * bytes_per_element
    size_in_mb = size_in_bytes / (1024 ** 2)  # Convert bytes to MB
    print("size in mb", size_in_mb)
    return size_in_mb


@torch.no_grad()
def evaluate(epoch, step, val_loader, val_feat_proc, detokenize,
             ema_model, loss_fn, greedy_decoder, use_amp):

    ema_model.eval()

    start_time = time.time()
    agg = {'losses': [], 'preds': [], 'txts': [], 'idx': []}
    logging.log_start(logging.constants.EVAL_START, metadata=dict(epoch_num=epoch))
    for i, batch in enumerate(val_loader):
        print(f'{val_loader.pipeline_type} evaluation: {i:>10}/{len(val_loader):<10}', end='\r')

        audio, audio_lens, txt, txt_lens = batch

        feats, feat_lens = val_feat_proc([audio, audio_lens])

        log_probs, log_prob_lens = ema_model(feats, feat_lens, txt, txt_lens)
        loss = loss_fn(log_probs[:, :log_prob_lens.max().item()],
                                  log_prob_lens, txt, txt_lens)

        pred = greedy_decoder.decode(ema_model, feats, feat_lens)

        agg['losses'] += helpers.gather_losses([loss.cpu()])
        agg['preds'] += helpers.gather_predictions([pred], detokenize)
        agg['txts'] += helpers.gather_transcripts([txt.cpu()], [txt_lens.cpu()], detokenize)

    wer, loss = process_evaluation_epoch(agg)

    logging.log_event(logging.constants.EVAL_ACCURACY, value=wer, metadata=dict(epoch_num=epoch))
    logging.log_end(logging.constants.EVAL_STOP, metadata=dict(epoch_num=epoch))

    log((epoch,), step, 'dev_ema', {'loss': loss, 'wer': 100.0 * wer, 'took': time.time() - start_time})
    ema_model.train()
    return wer



def main():
    logging.configure_logger('RNNT')
    logging.log_start(logging.constants.INIT_START)
    torch.cuda.empty_cache()

    
    duration = 0
    args = parse_args()
    print("dali device", args.dali_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    assert(torch.cuda.is_available())
    assert args.prediction_frequency is None or args.prediction_frequency % args.log_frequency == 0

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # set up distributed training
    multi_gpu = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        print_once(f'Distributed training with {world_size} GPUs\n')
    else:
        world_size = 1

    if args.seed is not None:
        logging.log_event(logging.constants.SEED, value=args.seed)
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)
        # np_rng is used for buckets generation, and needs the same seed on every worker
        np_rng = np.random.default_rng(seed=args.seed)

    init_log(args)

    cfg = config.load(args.model_config)
    config.apply_duration_flags(cfg, args.max_duration)

    assert args.grad_accumulation_steps >= 1
    assert args.batch_size % args.grad_accumulation_steps == 0, f'{args.batch_size} % {args.grad_accumulation_steps} != 0'
    logging.log_event(logging.constants.GRADIENT_ACCUMULATION_STEPS, value=args.grad_accumulation_steps)
    batch_size = args.batch_size // args.grad_accumulation_steps

    logging.log_event(logging.constants.SUBMISSION_BENCHMARK, value=logging.constants.RNNT)
    logging.log_event(logging.constants.SUBMISSION_ORG, value='my-organization')
    logging.log_event(logging.constants.SUBMISSION_DIVISION, value=logging.constants.CLOSED) # closed or open
    logging.log_event(logging.constants.SUBMISSION_STATUS, value=logging.constants.ONPREM) # on-prem/cloud/research
    logging.log_event(logging.constants.SUBMISSION_PLATFORM, value='my platform')

    logging.log_end(logging.constants.INIT_STOP)
    if multi_gpu:
        torch.distributed.barrier()
    logging.log_start(logging.constants.RUN_START)
    if multi_gpu:
        torch.distributed.barrier()

    print_once('Setting up datasets...')
    (
        train_dataset_kw,
        train_features_kw,
        train_splicing_kw,
        train_specaugm_kw,
    ) = config.input(cfg, 'train')
    (
        val_dataset_kw,
        val_features_kw,
        val_splicing_kw,
        val_specaugm_kw,
    ) = config.input(cfg, 'val')

    logging.log_event(logging.constants.DATA_TRAIN_MAX_DURATION,
                      value=train_dataset_kw['max_duration'])
    logging.log_event(logging.constants.DATA_SPEED_PERTURBATON_MAX,
                      value=train_dataset_kw['speed_perturbation']['max_rate'])
    logging.log_event(logging.constants.DATA_SPEED_PERTURBATON_MIN,
                      value=train_dataset_kw['speed_perturbation']['min_rate'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_N,
                      value=train_specaugm_kw['freq_masks'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_MIN,
                      value=train_specaugm_kw['min_freq'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_MAX,
                      value=train_specaugm_kw['max_freq'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_N,
                      value=train_specaugm_kw['time_masks'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_MIN,
                      value=train_specaugm_kw['min_time'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_MAX,
                      value=train_specaugm_kw['max_time'])
    logging.log_event(logging.constants.GLOBAL_BATCH_SIZE,
                      value=batch_size * world_size * args.grad_accumulation_steps)

    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)

    class PermuteAudio(torch.nn.Module):
        def forward(self, x):
            tmp = (x[0].permute(2, 0, 1), *x[1:])
            #print("Speedyloader: PermuteAudio = ", (time.time() - start) * 1000, "ms", " duration = ", duration)
            return tmp #(x[0].permute(2, 0, 1), *x[1:])

    train_augmentations = torch.nn.Sequential(
        train_specaugm_kw and features.SpecAugment(optim_level=args.amp, **train_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **train_splicing_kw),
        PermuteAudio(),
    )
    val_augmentations = torch.nn.Sequential(
        val_specaugm_kw and features.SpecAugment(optim_level=args.amp, **val_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **val_splicing_kw),
        PermuteAudio(),
    )

    logging.log_event(logging.constants.DATA_TRAIN_NUM_BUCKETS, value=args.num_buckets)
    train_dataset = AudioDataset(data_dir=args.dataset_dir, manifest_fpaths=args.train_manifests, tokenizer=tokenizer,  queue = queue, queue_timeout = queue_timeout,  rank=args.local_rank, normalize_transcripts=True)
    val_dataset = AudioDataset(data_dir=args.dataset_dir, manifest_fpaths=args.val_manifests, tokenizer=tokenizer,  queue = queue, queue_timeout = queue_timeout, rank=args.local_rank,  normalize_transcripts=True)

   

    if world_size > 1:
        sampler = DistributedSampler(train_dataset, shuffle=False)
    else:
        sampler = None

    num_workers_per_gpu = 12

    if sampler is not None:
        indices = list(sampler)
    else:
        indices = list(range(len(train_dataset)))
    
 
    producers = []

    indices_chunks = np.array_split(indices, num_workers_per_gpu)

    for i, producer_indices in enumerate(indices_chunks):
        print(f"GPU {args.local_rank}, Worker {i}: {len(producer_indices)} samples")

        producer = DataProducer(
            queue,
            train_dataset,
            list(producer_indices),  # Convert numpy array to list
            rank=args.local_rank
        )
        producer.start()
        producers.append(producer)


    

    steps_per_epoch = len(train_dataset) // (batch_size * args.grad_accumulation_steps)
    print("steps_per_epoch", steps_per_epoch)   
    for i in range(2):
        constructor_proc = Process(
            target=batch_constructor,
            args=(queue, queue_timeout, batch_queue, args.batch_size, stop_event)
        )
        constructor_proc.start()
 

    # logging.log_event(logging.constants.TRAIN_SAMPLES, value=len(train_loader))
    # logging.log_event(logging.constants.EVAL_SAMPLES, value=len(val_loader))

    # set up the model
    rnnt_config = config.rnnt(cfg)
    logging.log_event(logging.constants.MODEL_WEIGHTS_INITIALIZATION_SCALE, value=args.weights_init_scale)
    if args.weights_init_scale is not None:
        rnnt_config['weights_init_scale'] = args.weights_init_scale
    if args.hidden_hidden_bias_scale is not None:
        rnnt_config['hidden_hidden_bias_scale'] = args.hidden_hidden_bias_scale
    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
    model.to(device)
    blank_idx = tokenizer.num_labels
    loss_fn = RNNTLoss(blank_idx=blank_idx)
    logging.log_event(logging.constants.EVAL_MAX_PREDICTION_SYMBOLS, value=args.max_symbol_per_sample)
    greedy_decoder = RNNTGreedyDecoder( blank_idx=blank_idx,
                                        max_symbol_per_sample=args.max_symbol_per_sample)

    print_once(f'Model size: {num_weights(model) / 10**6:.1f}M params\n')

    opt_eps=1e-9
    logging.log_event(logging.constants.OPT_NAME, value='lamb')
    logging.log_event(logging.constants.OPT_BASE_LR, value=args.lr)
    logging.log_event(logging.constants.OPT_LAMB_EPSILON, value=opt_eps)
    logging.log_event(logging.constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=args.lr_exp_gamma)
    logging.log_event(logging.constants.OPT_LR_WARMUP_EPOCHS, value=args.warmup_epochs)
    logging.log_event(logging.constants.OPT_LAMB_LR_HOLD_EPOCHS, value=args.hold_epochs)
    logging.log_event(logging.constants.OPT_LAMB_BETA_1, value=args.beta1)
    logging.log_event(logging.constants.OPT_LAMB_BETA_2, value=args.beta2)
    logging.log_event(logging.constants.OPT_GRADIENT_CLIP_NORM, value=args.clip_norm)
    logging.log_event(logging.constants.OPT_LR_ALT_DECAY_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LR_ALT_WARMUP_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LAMB_LR_MIN, value=args.min_lr)
    logging.log_event(logging.constants.OPT_WEIGHT_DECAY, value=args.weight_decay)

    # optimization
    kw = {'params': model.param_groups(args.lr), 'lr': args.lr,
          'weight_decay': args.weight_decay}

    initial_lrs = [group['lr'] for group in kw['params']]

    print_once(f'Starting with LRs: {initial_lrs}')
    optimizer = FusedLAMB(betas=(args.beta1, args.beta2), eps=opt_eps, max_grad_norm=args.clip_norm, **kw)

    adjust_lr = lambda step, epoch: lr_policy(
        step, epoch, initial_lrs, optimizer, steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs, hold_epochs=args.hold_epochs,
        min_lr=args.min_lr, exp_gamma=args.lr_exp_gamma)

    if args.amp:
        model, optimizer = amp.initialize(
            models=model,
            optimizers=optimizer,
            opt_level='O1',
            max_loss_scale=512.0)

    if args.ema > 0:
        ema_model = copy.deepcopy(model).cuda()
    else:
        ema_model = None
    logging.log_event(logging.constants.MODEL_EVAL_EMA_FACTOR, value=args.ema)

    if multi_gpu:
        model = DistributedDataParallel(model)

    # load checkpoint
    meta = {'best_wer': 10**6, 'start_epoch': 0}
    checkpointer = Checkpointer(args.output_dir, 'RNN-T',
                                args.keep_milestones, args.amp)
    if args.resume:
        args.ckpt = checkpointer.last_checkpoint() or args.ckpt

    if args.ckpt is not None:
        checkpointer.load(args.ckpt, model, ema_model, optimizer, meta)

    start_epoch = meta['start_epoch']
    best_wer = meta['best_wer']
    last_wer = meta['best_wer']
    epoch = 1
    total_samples = 0
    batch_idx = 0  # Increment batch counter
    throughput_list = []
    throughput_list.append(args.local_rank)
    throughput_list.append("instant") 

    cum_avg_throughput_list =[]
    cum_avg_throughput_list.append(args.local_rank)
    start_training_time = time.time()


    step = start_epoch * steps_per_epoch + 1
    

    # training loop
    model.train()
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        
        print("\nSpeedyloader: epoch",epoch)

        logging.log_start(logging.constants.BLOCK_START,
                          metadata=dict(first_epoch_num=epoch,
                                        epoch_count=1))
        logging.log_start(logging.constants.EPOCH_START,
                          metadata=dict(epoch_num=epoch))

        epoch_utts = 0
        accumulated_batches = 0
        epoch_start_time = time.time()
        n = 0
        last_time = time.time()
        size = 0
        last_step =0
        next_batch =None

        while True:
            if batch_queue.empty():
                print("batch_queue is empty...")
                time.sleep(1)
                continue

            
            
            
            print(f"step: {n}, GPU: {args.local_rank}, Batch Queue size before pull:{batch_queue.qsize()}, Queue sizes: {queue.qsize()}, Timeout queue size: {queue_timeout.qsize()}")
            
            iteration_start_time = time.time()
            # Pull the next batch from the queue
            batch = batch_queue.get()
            if batch is None:
                print("Received sentinel. Exiting.")
                break
            print(f"step: {n}, GPU: {args.local_rank}, Batch Queue size after pull:{batch_queue.qsize()}, Queue sizes: {queue.qsize()}, Timeout queue size: {queue_timeout.qsize()}")
            audio, audio_lens, txt, txt_lens = collate_fn(batch)





            # Pin the memory of tensors
            audio, audio_lens, txt, txt_lens = (
                audio.pin_memory(), 
                audio_lens.pin_memory(), 
                txt.pin_memory(), 
                txt_lens.pin_memory()
            )

            # Asynchronously transfer data to GPU
            audio = audio.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            txt = txt.to(device, non_blocking=True)
            txt_lens = txt_lens.to(device, non_blocking=True)
            

    

            if accumulated_batches == 0:
                adjust_lr(step, epoch)
                optimizer.zero_grad()
                step_utts = 0
                step_start_time = time.time()
                all_feat_lens = []
            
    

        
            # Wait for the transfer stream to finish (synchronize if necessary)
            log_probs, log_prob_lens = model(audio, audio_lens, txt, txt_lens)
            print("log_probs_lens", log_prob_lens.shape)
            print("log_probs", log_probs.shape)


            print(f" Before slicing: txt.shape = {txt.shape}")  
            print(f"txt_lens.max().item() = {txt_lens.max().item()}")

            txt = txt[:, :txt_lens.max().item()].contiguous()

            print(f" After slicing: txt.shape = {txt.shape}") 
            
            print("running loss_fn")
        # Ensure both logits and labels are sliced to match the target length
            max_label_len = txt_lens.max().item()  # The maximum length of the labels in the batch

            # Slice logits to match the maximum label length
            log_probs_sliced = log_probs[:, :max_label_len, :].contiguous()

            # Slice the labels to match the maximum label length
            txt_sliced = txt[:, :max_label_len].contiguous()

            # Ensure the lengths are passed correctly
            loss = loss_fn(
                log_probs_sliced,  # Sliced log-probabilities (logits)
                log_prob_lens.contiguous(),  # Log-probability lengths (max length of logits)
                txt_sliced,  # Sliced ground truth sequences
                txt_lens.contiguous()  # Ground truth lengths (target sequence lengths)
            )

            size = size + calculate_tensor_size(audio) + calculate_tensor_size(audio_lens) + calculate_tensor_size(txt) + calculate_tensor_size(txt_lens)




            loss /= args.grad_accumulation_steps

            del log_probs, log_prob_lens

            if torch.isnan(loss).any():
                print_once(f'WARNING: loss is NaN; skipping update')
            else:
                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                loss_item = loss.item()
                del loss
                # step_utts += batch[0].size(0) * world_size
                # epoch_utts += batch[0].size(0) * world_size
                accumulated_batches += 1

            if accumulated_batches % args.grad_accumulation_steps == 0:

                total_norm = 0.0

                try:
                    if args.log_norm:
                        for p in getattr(model, 'module', model).parameters():
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                except AttributeError as e:
                    print_once(f'Exception happened: {e}')
                    total_norm = 0.0

            optimizer.step()


            iteration_duration1 = time.time() - iteration_start_time

            
                
            print('iteration duration', iteration_duration1)
            if time.time() - last_time >= 5:
                # Calculate number of iterations per second over the last 5 seconds
                iter_persec = (n - last_step) / (time.time() - last_time)
                throughput = size / (time.time() - last_time)
            
                
                with open(throughput_file, 'a', newline='') as f:
                    f.write(f"{n},{throughput},{time.time() - start_training_time},{time.time() - last_time}, {iter_persec}\n")
                

                last_time = time.time()
                last_step = n
                size = 0
        
        


            step += 1
            n+=1
            accumulated_batches = 0
            

        logging.log_end(logging.constants.EPOCH_STOP,
                        metadata=dict(epoch_num=epoch))

        epoch_time = time.time() - epoch_start_time
        log((epoch,), None, 'train_avg', {'throughput': epoch_utts / epoch_time,
                                          'took': epoch_time})

        if epoch % args.val_frequency == 0:
            wer = evaluate(epoch, step, val_loader, val_feat_proc,
                           tokenizer.detokenize, ema_model, loss_fn,
                           greedy_decoder, args.amp)

            last_wer = wer
            if wer < best_wer and epoch >= args.save_best_from:
                checkpointer.save(model, ema_model, optimizer, epoch,
                                  step, best_wer, is_best=True)
                best_wer = wer

        save_this_epoch = (args.save_frequency is not None and epoch % args.save_frequency == 0) \
                       or (epoch in args.keep_milestones)
        if save_this_epoch:
            checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)

        logging.log_end(logging.constants.BLOCK_STOP, metadata=dict(first_epoch_num=epoch))

        if last_wer <= args.target:
            logging.log_end(logging.constants.RUN_STOP, metadata={'status': 'success'})
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        if 0 < args.epochs_this_job <= epoch - start_epoch:
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        # end of epoch

    log((), None, 'train_avg', {'throughput': epoch_utts / epoch_time})

    if last_wer > args.target:
        logging.log_end(logging.constants.RUN_STOP, metadata={'status': 'aborted'})

    if epoch == args.epochs:
        evaluate(epoch, step, val_loader, val_feat_proc, tokenizer.detokenize,
                 ema_model, loss_fn, greedy_decoder, args.amp)

    flush_log()
    if args.save_at_the_end:
        checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)


if __name__ == "__main__":
    args = parse_args()

    import threading


    queue_timeout = Queue(100)
    batch_queue = Queue(100)

    queue = Queue(100)  
    stop_event = Event()
  

    import csv
    import threading
    import time


    # Initialize lists
    queue_sizes = []
    queue_timeout_sizes = []
    batch_queue_sizes = []
    queue_sizes.append(args.local_rank)
    queue_timeout_sizes.append(args.local_rank)
    batch_queue_sizes.append(args.local_rank)
    timestamps = []

    start_time = time.time()  # Set a reference start time
    file_path = 'queue_sizes_log.csv'



    # throughput_file = '/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_diff_#GPUs_speedy/results_10s/2gpu_throughput.csv'
    throughput_file = 'test_speedy3.csv'    
    with open(throughput_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Add a header if the file is empty (optional).
        if csvfile.tell() == 0:
            csv_writer.writerow(['iteration','throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec'])





    # # Open the CSV file globally (outside the thread) and pass the writer to the thread
    # csvfile = open(file_path, 'a', newline='')  # Open the file manually
    # csv_writer = csv.writer(csvfile)

    # # Add header to CSV file if it's empty
    # if csvfile.tell() == 0:
    #     csv_writer.writerow(['GPU', 'Timestamp', 'Queue Size', 'Queue Timeout Size', 'Batch Queue Size'])

 

    def track_queue_sizes():
        while not stop_event.is_set():
            current_time = time.time() - start_time  # Convert to relative seconds
            timestamps.append(round(current_time))  # Store whole seconds
            queue_sizes.append(queue.qsize())  # Store queue size
            queue_timeout_sizes.append(queue_timeout.qsize())  # Store timeout queue size
            batch_queue_sizes.append(batch_queue.qsize())  # Store batch queue size

            # Print the values to monitor
            print("timestamps", timestamps)
            print("queue_sizes", queue_sizes)
            print("queue_timeout_sizes", queue_timeout_sizes)
            print("batch_queue_sizes", batch_queue_sizes)

        
            # csv_writer.writerow([args.local_rank, round(current_time), queue.qsize(), queue_timeout.qsize(), batch_queue.qsize()])
            
            time.sleep(1)  # Log every 1 second

    # Start the queue monitoring thread
    monitor_thread = threading.Thread(target=track_queue_sizes, daemon=True)
    monitor_thread.start()

    # Call the main function (you may need to replace this with your main training logic)
    main()

    # Wait for the monitoring thread to finish
    monitor_thread.join()

  