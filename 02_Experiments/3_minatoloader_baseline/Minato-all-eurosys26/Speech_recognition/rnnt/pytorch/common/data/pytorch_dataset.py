# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##this is the dataset file that I can use to run pytorch datalaoder, DALI is the original one 
import math
import json
from pathlib import Path
import time
import numpy as np
import sox
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import multiprocessing

from common.audio import (audio_from_file, AudioSegment, SpeedPerturbation)
from common.text import _clean_text, punctuation_map

from common.helpers import print_once
from common.sampler import BucketingSampler
import numpy as np

def normalize_string(s, charset, punct_map):
    """Normalizes string.

    Example:
        'call me at 8:00 pm!' -> 'call me at eight zero pm'
    """
    charset = set(charset)
    try:
        text = _clean_text(s, ["english_cleaners"], punct_map).strip()
        return ''.join([tok for tok in text if all(t in charset for t in tok)])
    except:
        print(f"WARNING: Normalizing failed: {s}")
        return None



class FilelistDataset(Dataset):
    def __init__(self, filelist_fpath):
        self.samples = [line.strip() for line in open(filelist_fpath, 'r')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        print("Speedyloader: FilelistDataset.getitem")
        audio, audio_len = audio_from_file(self.samples[index])
        return (audio.squeeze(0), audio_len, torch.LongTensor([0]),
                torch.LongTensor([0]))


class SingleAudioDataset(FilelistDataset):
    def __init__(self, audio_fpath):
        self.samples = [audio_fpath]


def preprocess_audio1(waveform, n_fft=400, hop_length=160, n_mels=240):
    """
    Converts raw audio (waveform) into a spectrogram-like feature with 240 dimensions.
    Uses STFT to generate a frequency-time representation.
    
    Args:
    - waveform: (audio_length,) or (batch_size, audio_length)
    - n_fft: FFT window size
    - hop_length: Hop size (stride) for STFT
    - n_mels: Expected number of features (should match model expectation)

    Returns:
    - Processed audio tensor of shape (batch_size, time_steps, n_mels)
    """
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






class AudioDataset(Dataset):
    def __init__(self, data_dir, manifest_fpaths,
                 tokenizer,
                 sample_rate=16000, min_duration=0.1, max_duration=float("inf"),
                 max_utts=0, normalize_transcripts=True,
                 trim_silence=False,
                 speed_perturbation=None,
                 ignore_offline_speed_perturbation=False):
        """Loads audio, transcript and durations listed in a .json file."""

        self.data_dir = data_dir

        self.tokenizer = tokenizer
        self.punctuation_map = punctuation_map(self.tokenizer.charset)

        self.max_utts = max_utts
        self.normalize_transcripts = normalize_transcripts
        self.ignore_offline_speed_perturbation = ignore_offline_speed_perturbation

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.trim_silence = trim_silence
        self.sample_rate = sample_rate
        self.sample_count = 0  # To keep track of how many samples we've processed
        


        perturbations = []
        if speed_perturbation is not None:
            perturbations.append(SpeedPerturbation(**speed_perturbation))
        self.perturbations = perturbations

        self.max_duration = max_duration

        self.samples = []
        self.duration = 0.0
        self.duration_filtered = 0.0

        for fpath in manifest_fpaths:
            self._load_json_manifest(fpath)
    



    def __len__(self):
        return len(self.samples)


    def preprocess_audio(self, segment):
        self.sample_count += 1
        
        # Check if this sample needs heavy processing
        if self.sample_count % 5 == 0:
            print(f"Heavy processing added at sample {self.sample_count}!")
            self.heavy_processing(segment)
       
        else:
            # Regular light work processing
            self.light_processing(segment)

        return segment

    def light_processing(self, segment):
        """ Simulate light work worth ~0.5 seconds for regular samples """
        start_time = time.time()
        while time.time() - start_time < 0.5:
            _ = sum(i for i in range(10000))  # Simulate light computational work

        print(f"Light preprocessing time for sample {self.sample_count} completed.")

    def heavy_processing(self, segment):
        """ Simulate heavy work for every 5th sample """
        heavy_work_start = time.time()
        while time.time() - heavy_work_start < 3:
            _ = sum(i for i in range(1000))  # Simulate more intensive computational work

        print(f"Heavy work took {time.time() - heavy_work_start:.2f} seconds")

    
    def __getitem__(self, index):
        print("Speedyloader: AudioDataset.getitem")
        
        s = self.samples[index]
        rn_indx = np.random.randint(len(s['audio_filepath']))
        duration = s['audio_duration'][rn_indx] if 'audio_duration' in s else 0
        offset = s.get('offset', 0)

        # Load Audio
        segment = AudioSegment(
            s['audio_filepath'][rn_indx], target_sr=self.sample_rate,
            offset=offset, duration=duration, trim=self.trim_silence)
        segment = torch.FloatTensor(segment.samples)

        # Start the timer for preprocessing
        start_time = time.time()
        segment1= AudioSegment(
            s['audio_filepath'][rn_indx], target_sr=self.sample_rate,
            offset=offset, duration=duration, trim=self.trim_silence)
        segment1 = torch.FloatTensor(segment1.samples)

        # Preprocessing step that should take around 10ms
        segment_x = self.preprocess_audio(segment1)

        # Measure the time taken for the preprocessing
        preprocessing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        print(f"Preprocessing the audio took {preprocessing_time:.2f} ms")

        return (segment,
                torch.tensor(segment.shape[0]).int(),
                torch.tensor(s["transcript"]),
                torch.tensor(len(s["transcript"])).int())

    def _load_json_manifest(self, fpath):
        j = json.load(open(fpath, "r", encoding="utf-8"))
        for i, s in enumerate(j):
            if i % 1000 == 0:
                print(f'{i:>10}/{len(j):<10}', end='\r')

            s_max_duration = s['original_duration']

            s['duration'] = s.pop('original_duration')
            if not (self.min_duration <= s_max_duration <= self.max_duration):
                self.duration_filtered += s['duration']
                continue

            # Prune and normalize according to transcript
            tr = (s.get('transcript', None) or
                  self.load_transcript(s['text_filepath']))

            if not isinstance(tr, str):
                print(f'WARNING: Skipped sample (transcript not a str): {tr}.')
                self.duration_filtered += s['duration']
                continue

            if self.normalize_transcripts:
                tr = normalize_string(tr, self.tokenizer.charset, self.punctuation_map)

            s["transcript"] = self.tokenizer.tokenize(tr)

            files = s.pop('files')
            if self.ignore_offline_speed_perturbation:
                files = [f for f in files if f['speed'] == 1.0]

            s['audio_duration'] = [f['duration'] for f in files]
            s['audio_filepath'] = [str(Path(self.data_dir, f['fname']))
                                   for f in files]
            self.samples.append(s)
            self.duration += s['duration']

            if self.max_utts > 0 and len(self.samples) >= self.max_utts:
                print(f'Reached max_utts={self.max_utts}. Finished parsing {fpath}.')
                break

    def load_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return transcript


    def wait_for_active_processes(self):
        """ This function allows you to later wait for all active processes to finish if needed """
        for process in self.active_processes:
            process.join()



def collate_fn(batch):
    bs = len(batch)
    print("bs:", bs)
    max_len = lambda l, idx: max(el[idx].size(0) for el in l)
    audio = torch.zeros(bs, max_len(batch, 0))
    audio_lens = torch.zeros(bs, dtype=torch.int32)
    transcript = torch.zeros(bs, max_len(batch, 2))
    transcript_lens = torch.zeros(bs, dtype=torch.int32)

    for i, sample in enumerate(batch):
        audio[i].narrow(0, 0, sample[0].size(0)).copy_(sample[0])
        audio_lens[i] = sample[1]
        transcript[i].narrow(0, 0, sample[2].size(0)).copy_(sample[2])
        transcript_lens[i] = sample[3]
        print("Speedyloader: audio=", audio.shape)
    
    audio = preprocess_audio1(audio)
    print("audio after", audio.shape)
    print("transcript", transcript.shape)
    
    return audio, audio_lens, transcript, transcript_lens

        
# class AudioDataset1(Dataset):
#     def __init__(self, data_dir, manifest_fpaths,
#                  tokenizer,
#                  sample_rate=16000, min_duration=0.1, max_duration=float("inf"),
#                  max_utts=0, normalize_transcripts=True,
#                  trim_silence=False,
#                  speed_perturbation=None,
#                  ignore_offline_speed_perturbation=False):
#         """Loads audio, transcript and durations listed in a .json file.

#         Args:
#             data_dir: absolute path to dataset folder
#             manifest_filepath: relative path from dataset folder
#                 to manifest json as described above. Can be coma-separated paths.
#             tokenizer: class converting transcript to tokens
#             min_duration (int): skip audio shorter than threshold
#             max_duration (int): skip audio longer than threshold
#             max_utts (int): limit number of utterances
#             normalize_transcripts (bool): normalize transcript text
#             trim_silence (bool): trim leading and trailing silence from audio
#             ignore_offline_speed_perturbation (bool): use precomputed speed perturbation

#         Returns:
#             tuple of Tensors
#         """
#         self.data_dir = data_dir

#         self.tokenizer = tokenizer
#         self.punctuation_map = punctuation_map(self.tokenizer.charset)

#         self.max_utts = max_utts
#         self.normalize_transcripts = normalize_transcripts
#         self.ignore_offline_speed_perturbation = ignore_offline_speed_perturbation

#         self.min_duration = min_duration
#         self.max_duration = max_duration
#         self.trim_silence = trim_silence
#         self.sample_rate = sample_rate
#         self.sample_count = 0  # To keep track of how many samples we've processed

#         perturbations = []
#         if speed_perturbation is not None:
#             perturbations.append(SpeedPerturbation(**speed_perturbation))
#         self.perturbations = perturbations

#         self.max_duration = max_duration

#         self.samples = []
#         self.duration = 0.0
#         self.duration_filtered = 0.0

#         for fpath in manifest_fpaths:
#             self._load_json_manifest(fpath)


#     def preprocess_audio(self, segment):
#         # Increment sample count to track how many samples we've processed
#         self.sample_count += 1
        
#         # Simulate light work worth ~0.5 seconds for regular samples
#         start_time = time.time()
#         while time.time() - start_time < 0.5:
#             # Simulate some computational work (a small sum computation)
#             _ = sum(i for i in range(10000))  # Increase the range for more computation
        
#         # Every 5th sample, introduce heavier workload (simulate 3 seconds of processing)
#         if self.sample_count % 5 == 0:
#             print(f"Heavy processing added at sample {self.sample_count}!")
#             heavy_work_start = time.time()
#             while time.time() - heavy_work_start < 10:
#                 # Simulate more intensive computational work for 3 seconds
#                 _ = sum(i for i in range(1000))  # Increase range for more work
#             print(f"Heavy work took {time.time() - heavy_work_start:.2f} seconds")

#         print(f"Preprocessing time for sample {self.sample_count} completed.")
#         return segment



#     def __getitem__(self, index):
#         print("Speedyloader: AudioDataset.getitem")
        
#         s = self.samples[index]
#         rn_indx = np.random.randint(len(s['audio_filepath']))
#         duration = s['audio_duration'][rn_indx] if 'audio_duration' in s else 0
#         offset = s.get('offset', 0)

#         # Load Audio
#         segment = AudioSegment(
#             s['audio_filepath'][rn_indx], target_sr=self.sample_rate,
#             offset=offset, duration=duration, trim=self.trim_silence)
#         segment = torch.FloatTensor(segment.samples)

#         # Start the timer for preprocessing
#         start_time = time.time()
#         segment1= AudioSegment(
#             s['audio_filepath'][rn_indx], target_sr=self.sample_rate,
#             offset=offset, duration=duration, trim=self.trim_silence)
#         segment1 = torch.FloatTensor(segment1.samples)

#         # Preprocessing step that should take around 10ms
#         segment_x = self.preprocess_audio(segment1)

#         # Measure the time taken for the preprocessing
#         preprocessing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
#         print(f"Preprocessing the audio took {preprocessing_time:.2f} ms")

#         return (segment,
#                 torch.tensor(segment.shape[0]).int(),
#                 torch.tensor(s["transcript"]),
#                 torch.tensor(len(s["transcript"])).int())

#     def __len__(self):
#         return len(self.samples)
  

#     def _load_json_manifest(self, fpath):
#         j = json.load(open(fpath, "r", encoding="utf-8"))
#         for i, s in enumerate(j):
#             if i % 1000 == 0:
#                 print(f'{i:>10}/{len(j):<10}', end='\r')

#             s_max_duration = s['original_duration']

#             s['duration'] = s.pop('original_duration')
#             if not (self.min_duration <= s_max_duration <= self.max_duration):
#                 self.duration_filtered += s['duration']
#                 continue

#             # Prune and normalize according to transcript
#             tr = (s.get('transcript', None) or
#                   self.load_transcript(s['text_filepath']))

#             if not isinstance(tr, str):
#                 print(f'WARNING: Skipped sample (transcript not a str): {tr}.')
#                 self.duration_filtered += s['duration']
#                 continue

#             if self.normalize_transcripts:
#                 tr = normalize_string(tr, self.tokenizer.charset, self.punctuation_map)

#             s["transcript"] = self.tokenizer.tokenize(tr)

#             files = s.pop('files')
#             if self.ignore_offline_speed_perturbation:
#                 files = [f for f in files if f['speed'] == 1.0]

#             s['audio_duration'] = [f['duration'] for f in files]
#             s['audio_filepath'] = [str(Path(self.data_dir, f['fname']))
#                                    for f in files]
#             self.samples.append(s)
#             self.duration += s['duration']

#             if self.max_utts > 0 and len(self.samples) >= self.max_utts:
#                 print(f'Reached max_utts={self.max_utts}. Finished parsing {fpath}.')
#                 break

#     def load_transcript(self, transcript_path):
#         with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
#             transcript = transcript_file.read().replace('\n', '')
#         return transcript





def get_data_loader(dataset, batch_size, world_size, rank, shuffle=True,
                    drop_last=True, num_workers=24, num_buckets=None):
    if world_size != 1:
        loader_shuffle = False
        if num_buckets:
            assert shuffle, 'only random buckets are supported'
            sampler = BucketingSampler(
                dataset,
                batch_size,
                num_buckets,
                world_size,
                rank,
            )
            print('Using BucketingSampler')
        else:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            print('Using DistributedSampler')
    else:
        loader_shuffle = shuffle
        sampler = None
        print('Using no sampler')

    return DataLoader(
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=sampler,
        shuffle=loader_shuffle,
        dataset=dataset,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
