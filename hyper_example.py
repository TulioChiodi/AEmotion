from funcs import hyper

hyper.connect()

for i in range(1,4):
    topic_pub = f"topic/{i}"
    message = f"Message: {i}"
    hyper.send(topic_pub, message)
    # You can use hyper.message_feedback variable for the message logging (only lastone)

hyper.disconnect()


## envios assincronos, sincronicidade é necessária?
## Acoplar a rede
## Implementar sistema de parada (Tkinter?)

