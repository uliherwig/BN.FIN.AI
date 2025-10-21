from aiokafka import AIOKafkaConsumer
from datetime import datetime
import json

async def start_kafka_consumer():
    return
    consumer = AIOKafkaConsumer(
        "optimize",
        bootstrap_servers="localhost:9092",
        group_id="BN.PROJECT"
    )
    await consumer.start()
    try:
        async for msg in consumer:
            if msg.value is not None:
                data = json.loads(msg.value.decode("utf-8"))
                message_type = data.get("MessageType")
                if(message_type == "quotes"):
                    print(f"Received quotes for asset: {data.get('Asset')}")
            else:
                print("Received message with no value.")
                   
            if(message_type == "stopTest"):
                print("Received stoptest message, stopping consumer.")
            
            if(message_type != "quotes"):
                print(f"Received message: {message_type}")  
            
    finally:
        await consumer.stop()
