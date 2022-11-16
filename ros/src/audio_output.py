#!/usr/bin/env python3
import os
import numpy as np

import rospy
from soundfile import SoundFile

from jetson_voice import AudioOutput
from jetson_voice_ros.msg import Audio


class AudioOutput:
    def __init__(self, device_name, sample_rate, chunk_size):    # TODO: add default parameters    
        # create topics
        self.audio_subscriber = rospy.Subscriber('audio_out', Audio, self.audio_listener, 10)
        
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        if self.device_name == '':
            raise ValueError("must set the 'device' parameter to either an input audio device ID/name or the path to a .wav file")
        
        print(f'device={self.device_name}')
        print(f'sample_rate={self.sample_rate}')
        print(f'chunk_size={self.chunk_size}')
        
        # check if this is an audio device or a wav file
        file_ext = os.path.splitext(self.device_name)[1].lower()
        
        if file_ext == '.wav' or file_ext == '.wave':
            self.wav = SoundFile(self.device_name, mode='w', samplerate=self.sample_rate, channels=1)
            self.device = None
        else:
            self.wav = None
            self.device = AudioOutput(self.device_name, sample_rate=self.sample_rate, chunk_size=self.chunk_size)

    def audio_listener(self, msg):        
        if msg.info.sample_rate != self.sample_rate:
            print(f"audio has sample_rate {msg.info.sample_rate}, but audio device is using sample_rate {self.sample_rate}")
            
        samples = np.frombuffer(msg.data, dtype=msg.info.sample_format)
        
        print(f'recieved audio samples {samples.shape} dtype={samples.dtype}') # rms={np.sqrt(np.mean(samples**2))}')
        
        if self.device is not None:
            self.device.write(samples)
        else:
            self.wav.write(samples)

if __name__ == "__main__":
   pass
   # TODO