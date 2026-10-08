import subprocess
import threading
import time
import torch
import os
import numpy as np
from queue import Full, Empty
import multiprocessing as mp
import psutil
import csv
from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.data.collate_batch import BatchCollator
collator = BatchCollator(32)

# Global stop event for monitoring
data_stop_event = mp.Event()
class DataProducer(mp.Process):
    def __init__(self, queue, dataset, indices):
        super().__init__()
        self.queue = queue
        self.dataset = dataset       
        self.indices = indices
 
    
    def run(self):
        for idx in self.indices:
            self.process_sample(idx)

    def process_sample(self, idx):
        """Handles loading and queuing of a single sample."""
        try:
            print("len self.dataset", len(self.dataset))
            batch = self.dataset[idx]  # Fetch sample from dataset
            if batch is None or len(batch) < 1:
                return

            print("type of batch", type(batch))
            print("type of image", type(batch[0]))
            print("image shape", batch[0].shape)
            print("batch in producers", batch)
            
            print("get batch from self.dataset ", len(batch)) 

         
            while self.queue.qsize() >=  450:
                print(f"Main queue almost full (size: {self.queue.qsize()}). Pausing...")
                time.sleep(1)
            

            images, targets, idx = batch
          
            print("add to the queue", self.queue.qsize())
            
            self.queue.put(batch, block=True)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")

    def _stop(self):
        data_stop_event.set()


class AsynchronousLoader:
    def __init__(self, dataset, queue, queue_timeout, batch_queue, device, shards, sampler, batch_size, shuffle = False,  pin_memory=True, num_workers=1, rank=None):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.queue = queue
        self.batch_queue = batch_queue
        self.device = device
        self.shards = shards
        self.num_workers = num_workers
        self.rank = rank
        self.queue_timeout = queue_timeout
        self.epoch_batches = len(self.dataset) // (batch_size * self.shards)
        self.shuffle = shuffle
        self.indices = list(sampler)
       
        self.pin_memory = pin_memory
        
        self.start_batch_data_collection()
        print(f"GPU Rank {self.rank} is assigned {len(self.indices)} samples.")
        print(f"Created queue for GPU Rank {self.rank}.")
        self.start_threads()
    
    def __len__(self):
        return self.epoch_batches
    
    def start_batch_data_collection(self):
        processes = []
        for i in range(2):
            process = mp.Process(target=self.collect_batch_data)
            process.start()
            processes.append(process)
    
    def start_threads(self):
        self.producers = []
        total_samples = len(self.indices)
        indices_per_producer = max(1, total_samples // self.num_workers)
        start_idx = 0

        for i in range(self.num_workers):
            end_idx = min(start_idx + indices_per_producer, total_samples)
            producer_indices = self.indices[start_idx:end_idx]
            start_idx = end_idx

            if not producer_indices:
                break

            print(f"[GPU Rank {self.rank}] Producer {i} assigned indices: {producer_indices}")
            
            producer = DataProducer(self.queue, self.dataset, producer_indices)

            producer.start()
            self.producers.append(producer)

   

    def stop_threads(self):
        print('STOPPING THREADS ')
        data_stop_event.set()
        for producer in self.producers:
            if producer.is_alive():
                print(f"Joining producer")
                producer.join(timeout=5)
                if producer.is_alive():
                    print(f"Producer did not exit, forcefully terminating...")
                    producer._stop()
        
        self.queue.close()
        self.queue.join_thread()
        print("Queue closed and joined.")

    

    
    def collect_batch_data(self, save_folder="batch_logs"):
        # 1. Ensure the folder exists
        os.makedirs(save_folder, exist_ok=True)

        # 2. Create full path for CSV file inside that folder
        csv_filename = os.path.join(save_folder, f"batch_composition_rank{self.rank}.csv")

        file_exists = os.path.isfile(csv_filename)
        csv_file = open(csv_filename, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow(['batch_id', 'batch_composition'])

        batch_id = 0

        while True:
            batch_data = []
            batch_types = []

            while len(batch_data) < self.batch_size:
                if self.queue.empty() and self.queue_timeout.empty():
                    time.sleep(0.1)
                    continue

                if not self.queue.empty():
                    item = self.queue.get()
                    if item is None:
                        print("Received sentinel. Exiting training loop.")
                        csv_file.close()
                        return

                    image, target, idx = item
                    if not isinstance(image, torch.Tensor):
                        continue

                    batch_data.append(item)
                    batch_types.append('S')

                elif not self.queue_timeout.empty() and self.queue_timeout.qsize() > 1:
                    item = self.queue_timeout.get()
                    if item is None:
                        print("Received sentinel from timeout queue. Exiting training loop.")
                        csv_file.close()
                        return

                    image, target, idx = item
                    if not isinstance(image, torch.Tensor):
                        continue

                    batch_data.append(item)
                    batch_types.append('L')

                else:
                    if len(batch_data) < self.batch_size and self.queue.empty() and self.queue_timeout.empty():
                        time.sleep(0.01)

            if not batch_data:
                csv_file.close()
                return

            batch_data = collator(batch_data)

            if self.pin_memory:
                batch_data = self._move_to_pinned_memory(batch_data)

            # Write batch composition to CSV
            batch_comp_str = ' '.join(batch_types)
            csv_writer.writerow([batch_id, batch_comp_str])
            csv_file.flush()

            batch_id += 1

            while self.batch_queue.qsize() > 450:
                print(f"Batch queue almost full (size: {self.batch_queue.qsize()}). Pausing...")
                time.sleep(1)

            print(f"[Rank {self.rank}] Added batch {batch_id} with composition: {batch_comp_str} to batch queue")
            self.batch_queue.put(batch_data)

    def _move_to_pinned_memory(self, batch):
        pinned_batch = []
        image, target, idx = batch  # Unpack tuple
                    
                    # Pin image if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.pin_memory()
                    
        # Pin each tensor inside the target dictionary
        if isinstance(target, dict):
            target = {k: v.pin_memory() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                    
        pinned_batch.append((image, target, idx))  # Recreate tuple with pinned memory
        batch = (image, target, idx)
                    
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        """ Fetch preloaded batch from double buffer and transfer to GPU """
        batches = []
        while True:
            try:
                print("get two from the batch queue", self.batch_queue.qsize())
                for _ in range(2):  # Try to fetch 7 batches
                    batches.append(self.batch_queue.get(timeout=0.05))
                return tuple(batches)  


            except Empty:
                time.sleep(0.05)
            except Exception as e:
                print(f"Batch error: {e}")
                raise StopIteration



 # def collect_batch_data1(self):
    #     while True:
    #         batch_data = []

    #         while len(batch_data) < self.batch_size:
    #             print(f"Queue size: {self.queue.qsize()}, Timeout queue size: {self.queue_timeout.qsize()}")
                
    #             if self.queue.empty() and self.queue_timeout.empty():
    #                 time.sleep(0.1)
    #                 continue

    #             if not self.queue.empty():
    #                 item = self.queue.get()
    #                 print("item from queue", item[0])
    #                 if item is None:
    #                     print("Received sentinel. Exiting training loop.")
    #                     return

    #                 image, target, idx = item
    #                 if not isinstance(image, torch.Tensor):
    #                     print(f"Skipping: Image is not a tensor.")
    #                     continue
                    
    #                 batch_data.append(item)

    #             if not self.queue_timeout.empty() and self.queue_timeout.qsize() > 10:
    #                 item = self.queue_timeout.get()
    #                 print("item from queue timeout", item)
    #                 if item is None:
    #                     print("Received sentinel from timeout queue. Exiting training loop.")
    #                     return
                    
    #                 image, target, idx = item
    #                 if not isinstance(image, torch.Tensor):
    #                     print(f"Skipping: Image is not a tensor.")
    #                     continue

    #                 batch_data.append(item)

    #             if len(batch_data) < self.batch_size and self.queue.empty() and self.queue_timeout.empty():
    #                 print("Both queues are empty. Waiting for data...")
    #                 time.sleep(0.01)
            
    #         if not batch_data:
    #             print("No data fetched. Ending training loop.")
    #             return
            
    #         print(f"Batch data size: {len(batch_data)}")
    #         print(f"Queue size after pull: {self.queue.qsize()}, Timeout queue size after pull: {self.queue_timeout.qsize()}")
            
    #         batch_data = collator(batch_data)
    #         print("add to the batch queue", self.batch_queue.qsize())
    #         if self.batch_queue.qsize() > 90:
    #             time.sleep(0.5)

    #         self.batch_queue.put(batch_data)