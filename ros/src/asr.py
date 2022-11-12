#!/usr/bin/env python3
import os
import numpy as np
from abs import abstractclass
from multiprocessing import Process

import rospy
import pyloudnorm as pyln
import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
from soundfile import SoundFile

from jetson_voice_ros.msg import Audio
from jetson_voice_ros.srv import Asr
from std_msgs.msg import String


@abstractclass
class ASR:
    
    def transcribe_audio_path(self, audio_path, *kwargs):
        raise NotImplementedError

    def transcribe_audio(self, audio):
        raise NotImplementedError

    def transcribe_audio(self, audio):
        raise NotImplementedError

class WhisperROS(ASR):

    def __init__(self, model_name="base", sample_rate=16000):
        super().__init__(self, ASRNode)    
        
        self.sample_rate = sample_rate

        # create topics
        self.get_logger().info(f"creating topics")
        self.audio_subscriber = self.create_subscription(Audio, 'audio_in', self.audio_listener, 10)
        self.transcript_publisher = self.create_publisher(String, 'transcripts', 10)
        
        # get node parameters
        self.declare_parameter('model', 'model_name')
        self.model_name = str(self.get_parameter('model').value)
        self.get_logger().info(f'model = {self.model_name}')

        # load the ASR model
        self.get_logger().info(f"model '{self.model_name}' ready")
        
        self.asr = whisper.load_model(model_name)
        self.decoding_options = whisper.DecodingOptions(
            language="en", without_timestamps=True, beam_size=1)
        # warmup
        self.get_logger().info(f"running warmup")
        mel = whisper.log_mel_spectrogram(torch.empty(torch.Size([480000])).to("cuda"))
        self.asr.decode(mel, self.decoding_options)
        
        self.get_logger().info(f"model '{self.model_name}' ready") 

    def transcribe(audio, decoding_options=None):
        if not decoding_options:
            decoding_options = self.decoding_options

        audio = whisper.pad_or_trim(audio.flatten())
        audio = torch.from_numpy(audio).to("cuda")
        
        st = time.time()
        mel = whisper.log_mel_spectrogram(audio)
        result = self.stt.decode(mel, decoding_options)
        with open(f'{audio_path}.txt', 'a') as txt:
            txt.write(f'{self.__class__.__name__}: {result.text}')
        self.get_logger().debug(f"{result}")
        end = time.time()
        self.get_logger().debug(f"\ttook {end-st} seconds")
        
        msg = String()
        msg.data = result.text
        self.transcript_publisher.publish(msg)

        return result.text
        
    def audio_listener(self, msg):
        if msg.info.sample_rate != self.sample_rate:
            self.get_logger().warning(f"audio has sample_rate {msg.info.sample_rate}, "
                                      f"but ASR expects sample_rate {self.asr.sample_rate}")
            
        samples = np.frombuffer(msg.data, dtype=msg.info.sample_format)
        self.get_logger().debug(f'received audio samples {samples.shape} dtype={samples.dtype}') # rms={np.sqrt(np.mean(samples**2))}')
        
        self.transcribe(samples)

    def transcribe_audio_path(self, audio_path, *kwargs):
        print(f"Starting transcribing audio {audio_path}")
        try:
            audio = whisper.load_audio(file=audio_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Trascribing {audio_path}: {str(e)}")
            raise e
            return None

if __name__ == "__main__":
    asr = WhisperROS()
    
    def handler(req):
        print(req)
        transcription = asr.transcribe_audio_path(req.audio_path)  # TODO: refactor to use audio instead of audio path
        print(transcription)
        return sttResponse(
            transcription=transcription
        )
    rospy.init_node('whisper_asr')
    service = rospy.Service('voice/stt/whisper', stt_srv, handler)   
    
    rospy.spin()