import paho.mqtt.client as mqtt
import time

client_name='instance1'
broker_address="146.164.26.62" 
port = 2494
user = 'participants'
pswd = 'prp1nterac'

client = mqtt.Client(client_name)

message_feedback = dict()

def on_message(client, userdata, message):
    infos = {'message' : str(message.payload.decode("utf-8")),
            'topic' : message.topic,
            'qos' : message.qos,
            'retain_flag' : message.retain}

    message_feedback.update(infos)     

    # print("message received: " ,str(message.payload.decode("utf-8")))
    # print("message topic: ",message.topic)
    # print("message qos: ",message.qos)
    # print("message retain flag: ",message.retain)


def on_log(client, userdata, level, buf):
    print("log: ",buf)

def connect():
    client.on_message=on_message 
    client.username_pw_set(user, pswd)
    client.connect(broker_address, port)
    client.loop_start()
    

def send(topic_sub, topic_pub, message):
    
    client.subscribe(topic_sub)
    client.publish(topic_pub, message)
    time.sleep(2)
    print(message_feedback)
    

def disconnect():
    client.loop_stop()

# if __name__ == '__main__':

#     message = 'ON'
#     topic = 'hiper/tulio13'

#     assert message_feedback, "message_feedback dict is empty, check if you are getting any response."
#     assert message == message_feedback['message'], f"Message is not {message}, instead: {message_feedback['message']}"
#     assert topic == message_feedback['topic'], f"topic is not {topic}, instead: {message_feedback['topic']}"

#     print('All good :)')