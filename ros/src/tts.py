#!/usr/bin/env python3
import os
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
        # TODO
        
    def text_listener(self, msg):
        text = msg.data.strip()
        
        if len(text) == 0:
            return
            
        print(f"running TTS on '{text}'")
        
        samples = self.tts(text) # TODO
        samples = audio_to_int16(samples)
        
        # publish message
        msg = Audio()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.model_name

        msg.info.channels = 1
        msg.info.sample_rate = self.tts.sample_rate
        msg.info.sample_format = str(samples.dtype)
        
        msg.data = samples.tobytes()
        
        self.audio_publisher.publish(msg)

    def __call__(self, req):
        return self.text_listener(req)
        

def main(args=None):
    tts = TTSROS()
    
    def handler(req):
        print(req)
        tts(req)  # TODO: refactor to use audio instead of audio path
        return TtsResponse()
    rospy.init_node('tts')
    service = rospy.Service('voice/tts', Tts, handler)   
    
    rospy.spin()