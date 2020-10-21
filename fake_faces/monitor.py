import threading
import psutil
import time
import os

def is_running(script):
    for q in psutil.process_iter():
        if q.name().startswith('python'):
            if len(q.cmdline())>1 and script in q.cmdline()[1] and q.pid !=os.getpid():
                return True

    return False

TIME_LIMIT = 36000
start = time.time()
while(True):
    if(time.time() - start >= TIME_LIMIT):
        break
    time.sleep(5) # query every 5 seconds

    if(is_running("train.py")):
        continue

    if(not is_running("train.py")):
        print('[INFO] Starting a new thread in 5 seconds ... ')
        time.sleep(5) # rest 5 seconds before new thread
        print('[INFO] New thread started, number of active thread : %d' % threading.active_count())
        t = threading.Thread(target=os.system, args=("python3 train.py --data_dir ../../datasets/CASIA-WebFace --num_images 10000 --epochs 500",))
        t.daemon = True
        t.start()
