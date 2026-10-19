import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import threading
import time
import psutil
import numpy as np
from queue import Empty, Full
from runtime.distributed_utils import get_rank, get_world_size

class DynamicDataProducer(mp.Process):
    def __init__(self, data_queue, indices_queue, dataset, stop_event, queue_size):
        super().__init__()
        self.data_queue = data_queue
        self.indices_queue = indices_queue
        self.dataset = dataset
        self.stop_event = stop_event
        self.queue_size = queue_size
        self.rank = get_rank()

    def run(self):
        while not self.stop_event.is_set():
            # Wait if data queue is full
            if self.data_queue.qsize() >= self.queue_size - 1:
                time.sleep(0.01)
                continue
                
            try:
                # Fetch next index to process
                idx = self.indices_queue.get(timeout=0.05)
            except Empty:
                # No more indices for this epoch
                break
                
            if self.stop_event.is_set():
                break
                
            sample = self.dataset[idx]
            if sample is not None:
                try:
                    self.data_queue.put(sample, timeout=1.0)
                except Full:
                    # If queue is full, we might lose the sample in this simple implementation,
                    # but we check qsize before, so it's unlikely.
                    pass

class WorkerScheduler(threading.Thread):
    def __init__(self, loader, max_workers, alpha=2.0, beta=2.0, theta_c=0.7):
        super().__init__()
        self.loader = loader
        self.max_workers = max_workers
        self.alpha = alpha
        self.beta = beta
        self.theta_c = theta_c
        self.stop_event = threading.Event()
        self.daemon = True
        
        # Initialize CPU utilization
        psutil.cpu_percent(interval=None)

    def run(self):
        while not self.stop_event.is_set():
            time.sleep(1.0) # Evaluate every second
            
            if self.stop_event.is_set():
                break
                
            q_size = self.loader.queue.qsize()
            c_usage = psutil.cpu_percent(interval=None) / 100.0
            
            # Formula 2
            delta_float = self.alpha * (1 - q_size / self.loader.queue_size) + self.beta * (c_usage - self.theta_c)
            # Clip between -2 and 2
            delta = int(np.clip(round(delta_float), -2, 2))
            
            current_workers = len(self.loader.producers)
            # Formula 1
            desired_workers = min(self.max_workers, max(1, current_workers + delta))
            
            if desired_workers > current_workers:
                self.loader.add_workers(desired_workers - current_workers)
                #print(f"[Scheduler] Queue: {q_size}/{self.loader.queue_size}, CPU: {c_usage:.2f}. Workers: {current_workers} -> {desired_workers}")
            elif desired_workers < current_workers:
                self.loader.remove_workers(current_workers - desired_workers)
                #print(f"[Scheduler] Queue: {q_size}/{self.loader.queue_size}, CPU: {c_usage:.2f}. Workers: {current_workers} -> {desired_workers}")

    def stop(self):
        self.stop_event.set()

class ScheduledAsynchronousLoader(DataLoader):
    def __init__(self, dataset, device, shards, rank, slow_processed_queue, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, max_workers=8, queue_size=50, drop_last=True, sampler=None):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=0, drop_last=drop_last, sampler=sampler)
        self.queue_size = queue_size
        self.device = device
        self.shards = shards
        self.initial_workers = num_workers
        self.max_workers = max_workers
        self.world_size = get_world_size()
        self.rank = rank
        self.slow_processed_queue = slow_processed_queue
        
        self.epoch_batches = len(self.dataset) // (self.batch_size * self.world_size)
        self.shuffle = shuffle
        
        self.queue = None
        self.indices_queue = None
        self.stop_event = None
        
        self.producers = []
        self.scheduler = None

    def add_workers(self, n):
        for _ in range(n):
            if len(self.producers) >= self.max_workers:
                break
            producer = DynamicDataProducer(self.queue, self.indices_queue, self.dataset, self.stop_event, self.queue_size)
            producer.start()
            self.producers.append(producer)

    def remove_workers(self, n):
        # We can't cleanly stop specific workers dynamically without a targeted stop event, 
        # so we just let them finish or terminate them.
        # Since we use daemon processes or we can just send a sentinel to indices_queue or terminate.
        for _ in range(n):
            if len(self.producers) <= 1:
                break # keep at least 1
            producer = self.producers.pop()
            producer.terminate()
            producer.join()

    def start_epoch(self):
        # Stop existing threads
        self.stop_epoch()
        
        # Recreate queues to avoid multiprocessing deadlock from terminated producers
        self.queue = mp.Queue(maxsize=self.queue_size)
        self.indices_queue = mp.Queue()
        self.stop_event = mp.Event()
        
        self.stop_event.clear()
        
        # Populate indices queue
        indices = list(self.sampler)
        for idx in indices:
            self.indices_queue.put(idx)
            
        # Start initial workers
        self.producers = []
        self.add_workers(self.initial_workers)
        
        # Start scheduler
        self.scheduler = WorkerScheduler(self, self.max_workers)
        self.scheduler.start()

    def stop_epoch(self):
        if self.scheduler is not None:
            self.scheduler.stop()
            self.scheduler.join()
            self.scheduler = None
            
        if self.stop_event is not None:
            self.stop_event.set()
        
        # Empty indices queue
        if self.indices_queue is not None:
            while not self.indices_queue.empty():
                try:
                    self.indices_queue.get_nowait()
                except Empty:
                    break
                
        # Terminate producers
        for producer in self.producers:
            if producer.is_alive():
                producer.terminate()
                producer.join(timeout=1)
        self.producers = []
        
    def __iter__(self):
        self.batches_processed = 0
        self.start_epoch()
        return self

    def __next__(self):
        if self.batches_processed >= self.epoch_batches:
            self.stop_epoch()
            raise StopIteration

        batch = []
        while len(batch) < self.batch_size:
            sample = None
            
            # 1. Always try fast queue
            try:
                sample = self.queue.get(timeout=0.05)
                if sample is not None:
                    batch.append(sample)
            except Empty:
                pass
                
            # 2. Check if slow queue has items
            if not self.slow_processed_queue.empty() and len(batch) < self.batch_size:
                try:
                    slow_sample = self.slow_processed_queue.get(timeout=0.05)
                    if slow_sample is not None:
                        batch.append(slow_sample)
                except Empty:
                    pass
                    
            # 3. Handle idle
            if sample is None and self.slow_processed_queue.qsize() <= 5:
                if self.stop_event.is_set() and self.indices_queue.empty():
                    if not batch:
                        raise StopIteration
                    else:
                        break

        images, labels = zip(*batch)
        images = [torch.from_numpy(img) for img in images]
        labels = [torch.from_numpy(lbl) if isinstance(lbl, np.ndarray) else lbl for lbl in labels]
        
        batch_images_tensor = torch.stack(images)
        batch_labels_tensor = torch.stack(labels)
        
        if batch_images_tensor.size(0) == self.batch_size:
            self.batches_processed += 1
            return batch_images_tensor, batch_labels_tensor
        else:
            return self.__next__()
