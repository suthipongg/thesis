import subprocess
import threading
import time
import torch
import os
import numpy as np
from queue import Full, Empty
import multiprocessing as mp
from torch.utils.data import DataLoader
import psutil
import csv


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

# Global stop event for monitoring

import torch

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



data_stop_event = mp.Event()
class DataProducer(mp.Process):
    def __init__(self, queue, dataset, batch_size, shuffle, pin_memory, device, indices, sampler, num_workers,rank):
        super().__init__()
        self.queue = queue
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.device = device
        self.indices = indices
        self.sampler = sampler
        self.num_workers = num_workers
        self.rank = rank
    



    def run(self):
        for idx in self.indices:
            
                self.process_sample(idx)

    def process_sample(self, idx):
        """Handles loading and queuing of a single sample."""
        try:
            batch = self.dataset[idx]  # Fetch sample from dataset
            print("batch", batch)

            # # Validate batch before enqueuing
            # if not batch or (isinstance(batch[0], str) and batch[0] == "timeout"):
            #       return  # Skip invalid data

            # If the queue is full, wait until it has space
            while self.queue.qsize() >= 100:  # Maximum queue size
                print(f"Queue full (size: {self.queue.qsize()}). Pausing...")
                time.sleep(2)  # Wait for 0.5 seconds before checking again

            # Wait until the queue size is below the threshold before adding
            while self.queue.qsize() > 90:
                time.sleep(0.5)  # Small sleep to allow processing

            # Enqueue the batch
            print("add sample to the queeu", self.queue.qsize())
            self.queue.put(batch)  # Blocks if queue is full

        except Exception as e:
            print(f"Error processing index {idx}: {e}")  # Log error

        






    def _stop(self):
        data_stop_event.set()
       


class AsynchronousLoader():
    def __init__(self, dataset, queue, queue_timeout, device, shards, rank, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, sampler=None):
        """
        Custom Asynchronous DataLoader.

        Args:
        - dataset: Dataset to load (must be first argument).
        - queue: Main queue for loading data.
        - queue_timeout: Queue for slow-processing samples.
        - device: Target device (CPU/GPU).
        - shards: Number of distributed nodes.
        - rank: GPU rank.
        - batch_size: Number of samples per batch.
        - shuffle: Shuffle data or not.
        - pin_memory: Use pinned memory.
        - num_workers: Number of workers (should be 0 for async loading).
        - sampler: Custom sampler for distributed training.
        """
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        
        
        self.queue = queue
        self.dataset   = dataset
      
        self.device = device
        self.shards = shards
        self.num_workers = num_workers
        self.rank = rank
        self.queue_timeout = queue_timeout
        self.batch_size = batch_size
        self.sampler = sampler

        self.epoch_batches = len(self.dataset) // (self.batch_size * self.shards)  # Total batches per epoch
        
        self.shuffle = shuffle

        self.indices = list(self.sampler) # it gives the indices per gpu
        print(f"GPU Rank {self.rank} is assigned {len(self.indices)} samples.")
        
        print(f"Created queue for GPU Rank {self.rank}.")

        self.start_threads()
    def __len__(self):
        return len(self.dataset) // (self.batch_size * self.shards)

    def start_threads(self):
        self.producers = []
        total_samples = len(self.indices)
        indices_per_producer = max(1, total_samples // self.num_workers)  # Ensure at least one index per producer
        start_idx = 0

        for i in range(self.num_workers):
            end_idx = min(start_idx + indices_per_producer, total_samples)
            producer_indices = self.indices[start_idx:end_idx]
            start_idx = end_idx

            if not producer_indices:  # Break if no indices are available
                break

            print(f"[GPU Rank {self.rank}] Producer {i} assigned indices: {producer_indices}")
            
            producer = DataProducer(self.queue, self.dataset, self.batch_size, self.shuffle,
                                    self.pin_memory, self.device, 
                                    producer_indices, self.sampler, self.num_workers,self.rank)

            producer.start()
            self.producers.append(producer)






    def stop_threads(self):
        print('STOPPING THREADS ')
        data_stop_event.set()  # Signal threads to stop
        for producer in self.producers:
            if producer.is_alive():
                print(f"Joining producer")
                producer.join(timeout=5)  # Add a timeout to avoid hanging forever
                if producer.is_alive():
                    print(f"Producer did not exit, forcefully terminating...")
                    producer._stop()  # Force stop if the producer doesn't exit cleanly (not recommended, but an option)


        # Clean up the multiprocessing queue if using multiprocessing.Queue
        self.queue.close()
        self.queue.join_thread()
        print("Queue closed and joined.")

    def __iter__(self):
        self.batches_processed = 0  # Reset batch counter for the new epoch
        return self
    def __next__(self):
            while True:
                batch_data = []

                while len(batch_data) < self.batch_size:  # Try to collect a full batch
                    print(f"Queue size: {self.queue.qsize()}, Timeout queue size: {self.queue_timeout.qsize()}")
               

                    # Fetch from the main queue
                    if not self.queue.empty():
                        item = self.queue.get()
                        print("item from queue",item[0])
                        if item is None:  
                            print("Received sentinel. Exiting training loop.")
                            raise StopIteration  # Stop iteration properly
                        batch_data.append(item)

                    # # Fetch from the timeout queue
                    if not self.queue_timeout.empty():
                        print("Queue timeout is not empty... getting sample from queue")
                       
                        item = self.queue_timeout.get()
                        print("item", item)
                        print("item from queue timeout",item[0])

                        if item is None:  
                            print("Received sentinel from timeout queue. Exiting training loop.")
                            raise StopIteration  # Stop iteration properly
    
      
                        batch_data.append(item)
                    

                
                batch_data = collate_fn(batch_data)

                if not batch_data:
                    print("No data fetched. Ending training loop.")
                    raise StopIteration  # Stop iteration properly

                print(f"Batch data size: {len(batch_data)}")
                print(f"Queue size after pull: {self.queue.qsize()}, Timeout queue size after pull: {self.queue_timeout.qsize()}")

                return batch_data  
