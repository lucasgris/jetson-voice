#!/usr/bin/env python3
import os
import datetime
import rospy
import numpy as np

from std_msgs.msg import String

from jetson_voice.utils import audio_to_int16
from jetson_voice_ros.msg import Audio
from src.srv import Tts, TtsResponse


@abstractclass
class TTS:
    pass

class TTSROS:
    def __init__(self):        
        # create topics
        self.text_subscriber = rospy.Subscriber(String, 'tts_text', self.text_listener, 10)
        self.audio_publisher = rospy.Publisher(Audio, 'tts_audio', 10)
        
        self.tacotron2 = torch.load("../data/nvidia_tacotron2pyt_fp16.pt")
        self.tacotron2 = self.tacotron2.to('cuda')
        self.tacotron2.eval()

        self.waveglow = torch.load("../data/nvidia_waveglowpyt_fp16_20210323.pt')
        self.waveglow = self.waveglow.to('cuda')
        self.waveglow.eval()

        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
        sequences, lengths = utils.prepare_input_sequence(["warmup" for _ in range(5)])

        start_time = time.time() 

        with torch.no_grad():
            mel, _, _ = tacotron2.infer(sequences, lengths)
            audio = waveglow.infer(mel)
        audio_numpy = audio[0].data.cuda().numpy()
        self.rate = 22050

        write(f"warmup.wav", self.rate, audio_numpy)

        end_time = time.time()

        print(f"Elapsed time: {end_time - start_time}")
        
    def text_listener(self, msg):
        text = msg.data.strip()
        
        if len(text) == 0:
            return
            
        print(f"running TTS on '{text}'")
        
        sequences, lengths = utils.prepare_input_sequence([text])

        start_time = time.time() 

        with torch.no_grad():
            mel, _, _ = tacotron2.infer(sequences, lengths)
            audio = waveglow.infer(mel)
        audio_numpy = audio[0].data.cuda().numpy()
        self.rate = 22050

        audio_path = f"{datetime.datetime.now()}.wav"
        write(audio_path, self.rate, audio_numpy)

        end_time = time.time()

        print(f"Elapsed time: {end_time - start_time}")
        
        samples = audio_to_int16(audio_numpy)
        
        # publish message
        msg = Audio()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.model_name

        msg.info.channels = 1
        msg.info.sample_rate = self.tts.sample_rate
        msg.info.sample_format = str(samples.dtype)
        
        msg.data = samples.tobytes()
        
        self.audio_publisher.publish(msg)

        return audio_path

    def __call__(self, req):
        return self.text_listener(req)
        

def main(args=None):
    tts = TTSROS()
    
    def handler(req):
        print(req)
        tts(req) 
        return TtsResponse()
    rospy.init_node('tts')
    service = rospy.Service('voice/tts', Tts, handler)   
    
    rospy.spin()