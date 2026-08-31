import boto3
import json
import logging
import os
import psycopg2 # Use psycopg2 for postgres as standard since we are in python container for postgresql
from jax_simulation import run_handover_simulation  # Bloco 470

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sqs = boto3.client('sqs', region_name='us-east-1')
queue_url = 'https://sqs.us-east-1.amazonaws.com/123456789012/handover-queue'

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "arkhe"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

def update_database(handover_id, result):
    conn = get_db_connection()
    if conn is None:
        logger.error(f"Cannot update database for handover {handover_id} due to connection error.")
        return False

    try:
        cur = conn.cursor()
        # Ensure result is JSON serialized
        metadata = json.dumps(result) if isinstance(result, dict) else result

        # We assume table handovers exists and we just update metadata or stability_index based on simulation
        cur.execute(
            "UPDATE handovers SET metadata = %s WHERE id = %s",
            (metadata, handover_id)
        )
        conn.commit()
        cur.close()
        logger.info(f"Database updated successfully for handover {handover_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating database for handover {handover_id}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def main():
    logger.info(f"Starting SQS worker for queue: {queue_url}")
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10
            )
            for msg in response.get('Messages', []):
                try:
                    body = json.loads(msg['Body'])
                    handover_id = body.get('handoverId')

                    if not handover_id:
                        logger.warning(f"Received message without handoverId: {body}")
                        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg['ReceiptHandle'])
                        continue

                    logger.info(f"Processing handover {handover_id}")

                    # Executa simulação JAX + LLM
                    result = run_handover_simulation(handover_id)

                    # Atualiza banco de dados (PostgreSQL) com resultado
                    success = update_database(handover_id, result)

                    if success:
                        # Se necessário, dispara novo evento SQS para downstream
                        sqs.send_message(
                            QueueUrl=queue_url,
                            MessageBody=json.dumps({'type': 'HANDOVER_PROCESSED', 'handoverId': handover_id})
                        )
                        logger.info(f"Successfully processed and updated handover {handover_id}")
                    else:
                        logger.warning(f"Simulation ran but database update failed for handover {handover_id}")

                    # Delete message whether it fully succeeded or we failed to update DB (could dead-letter here instead)
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg['ReceiptHandle'])
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode message body: {msg['Body']}")
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg['ReceiptHandle'])
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error receiving messages from SQS: {e}")
            import time
            time.sleep(5)

if __name__ == "__main__":
    main()
