#!/usr/bin/env python3
import os
import numpy as np

from jetson_voice.utils import AudioInput, audio_to_int16
from src.msg import Audio


class AudioInput:
    def __init__(self, 
                 device_name, 
                 sample_rate, 
                 chunk_size, 
                 resets):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.resets = resets
        
        # create topics
        self.audio_publisher = rospy.Publisher('audio_in', Audio, 10)
        
        self.reset_count = 0
        
        if self.device_name == '':
            raise ValueError("must set the 'device' parameter to either an input audio device ID/name or the path to a .wav file")
        
        print(f'device={self.device_name}')
        print(f'sample_rate={self.sample_rate}')
        print(f'chunk_size={self.chunk_size}')
        print(f'resets={self.resets}')
        
        # check if this is an audio device or a wav file
        file_ext = os.path.splitext(self.device_name)[1].lower()
        
        if file_ext == '.wav' or file_ext == '.wave':
            wav = self.device_name
            mic = ''
        else:
            wav = ''
            mic = self.device_name

        # create audio device
        self.device = AudioInput(wav=wav, mic=mic, sample_rate=self.sample_rate, chunk_size=self.chunk_size)
        self.device.open()

        # create a timer to check for audio samples
        self.timer = self.create_timer(self.chunk_size / self.sample_rate * 0.75, self.publish_audio)
        
    def publish_audio(self):
    
        while True:
            samples = self.device.next()
            
            if samples is not None:
                break
                
            print('no audio samples were returned from the audio device')
            
            if self.resets < 0 or self.reset_count < self.resets:
                self.reset_count += 1
                print(f'resetting audio device {self.device_name} (attempt {self.reset_count} of {self.resets})')
                self.device.reset()
            else:
                print(f'maximum audio device resets has been reached ({self.resets})')
                return
                
        if samples.dtype == np.float32:  # convert to int16 to make the message smaller
            samples = audio_to_int16(samples)

        if samples.dtype != np.int16:  # the other voice nodes expect int16/float32
            raise ValueError(f'audio samples are expected to have datatype int16, but they were {samples.dtype}')
        
        print(f'publishing audio samples {samples.shape} dtype={samples.dtype}') # rms={np.sqrt(np.mean(samples**2))}')
        
        # publish message
        msg = Audio()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.device_name

        msg.info.channels = 1  # AudioInput is set to mono
        msg.info.sample_rate = self.sample_rate
        msg.info.sample_format = str(samples.dtype)
        
        msg.data = samples.tobytes()
        
        self.audio_publisher.publish(msg)
        
if __name__ == "__main__":
   pass
   # TODO