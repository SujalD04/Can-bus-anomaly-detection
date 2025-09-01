import can
import pandas as pd
import time

def replay_log(channel='vcan_bus', log_file_path='path/to/your/attack_file.csv'):
    """
    Reads a CSV log file and replays the CAN messages onto a virtual bus.
    """
    print(f"Replaying messages from {log_file_path} on virtual bus '{channel}'...")
    
    try:
        df = pd.read_csv(log_file_path)
        # Ensure column names match your CSV file
        df = df.rename(columns={'arbitration_id': 'CAN_ID', 'data_field': 'Payload'})
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Connect to the same virtual bus as the classifier
    bus = can.interface.Bus(channel=channel, bustype='virtual')

    start_time = time.time()
    log_start_time = df['Timestamp'].iloc[0]

    for _, row in df.iterrows():
        # Calculate elapsed time to replay messages at the correct speed
        elapsed_time = time.time() - start_time
        log_elapsed_time = row['Timestamp'] - log_start_time
        
        if log_elapsed_time > elapsed_time:
            time.sleep(log_elapsed_time - elapsed_time)
            
        try:
            # Create and send the CAN message
            msg = can.Message(
                arbitration_id=int(row['CAN_ID'], 16),
                data=bytes.fromhex(row['Payload']),
                is_extended_id=False
            )
            bus.send(msg)
        except Exception as e:
            # Handles potential malformed data in the CSV
            pass # print(f"Skipping malformed row: {row}. Error: {e}")

    print("Replay complete.")
    bus.shutdown()

if __name__ == '__main__':
    # IMPORTANT: Update this path to an attack CSV from your dataset
    # e.g., './2011-chevrolet-impala/DoS_attacks/DoS_attack_20_percent.csv'
    replay_log(log_file_path='path/to/your/attack_file.csv')