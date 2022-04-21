"""
© Technology Innovation Institute. All rights reserved.
Author(s): saurabh.pathak@tii.ae
Created: 

Purpose:
"""
import concurrent.futures
import subprocess

import paramiko


def get_ip(executor, server):
    dead = subprocess.call(['ping', server], stdout=subprocess.DEVNULL)
    if not dead:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        try:
            client.connect(server, username='mluser', password='ssrc@2022', timeout=5)
            print('success with', server, flush=True)
            executor.shutdown(wait=False)
            exit(0)
        except Exception as e:
            print(server, e, flush=True)


def main():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=250)
    [executor.submit(get_ip, executor, f'10.161.2.{n}') for n in range(0, 256)]


if __name__ == '__main__':
    main()
