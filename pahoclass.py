import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe


class paho_client:
    def __init__(self, user, password, host, port):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.client = mqtt.Client(client_id="AEmotion", clean_session=False) #create new instance
        self.client.username_pw_set(user, password)
        print("Connecting to broker", host)
        self.client.connect(host, port, 60)

        # Callback Function on RECEIVING the subscribed Topic/Message
        def on_message(client, userdata, message):
            print("message received ", str(message.payload.decode("utf-8")))
            print("message topic=",message.topic)
            # print("message qos=",message.qos)
            # print("message retain flag=",message.retain)

        # Callback Function on PUBLISHING to the subscribed Topic/Message
        def on_log(client, userdata, level, buf):
            print("log: ", buf)

        self.client.on_message = on_message #attach function to callback
        self.client.on_log = on_log
        
    # # Subscribe to listen 
    # def topic_subscribe(self, topic):
    #     self.topic = topic
    #     self.client.loop_start() #start the loop
    #     self.client.subscribe(topic)
    #     print("Subscribed to: ",topic)


    def publish_single(self, message, topic=None):
        if topic is not None:
            self.client.publish(topic, message)
        else:
            self.client.publish(self.topic, message)

    def listen(self, topics):
        self.topics = topics # lista
        def on_message_print(client, userdata, message):
            print("%s %s" % (message.topic, message.payload))
        subscribe.callback(on_message_print, self.topics, hostname=self.host)











# %%
# # Callback Function on Connection with MQTT Server
# def on_connect( client, userdata, flags, rc):
#     print ("Connected with Code :" +str(rc))
#     # Subscribe Topic from here
#     client.subscribe("hiper/davitulio13")

# # Callback Function on Receiving the Subscribed Topic/Message

# client =  mqtt.Client(client_id="Davi", clean_session=True, userdata=None)
# client.on_connect = on_connect
# client.on_message = on_message

# client.username_pw_set("participants", "prp1nterac")
# client.connect("146.164.26.62", 2494, 60)

# client.loop_forever()   
