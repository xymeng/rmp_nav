# Multiple load-balancers that are useful for serving requests for multiple maps concurrently.
# The client must send a multi-part message [identity, request] where identity is a unique
# identifier for a particular load-balancer. For example, it can be a map name.
# The identities should be specified in the configuration file.

from __future__ import print_function
import yaml
import sys
import subprocess
import shlex
import zmq
import os
import copy
import time


def start_generators(generator_cmds, working_dir, env, addr):
    procs = []

    print(env)

    os_env = copy.deepcopy(os.environ)
    os_env.update(env)

    for cmd in generator_cmds:
        cmd_env = copy.deepcopy(os_env)
        if isinstance(cmd, dict):
            cmd_env.update(cmd['env'])
            cmd = cmd['cmd']

        cmd += ' --addr %s' % addr
        print('starting generator %s at %s' % (cmd, addr))
        procs.append(subprocess.Popen(
            shlex.split(cmd), env=cmd_env, cwd=os.path.realpath(working_dir)))

    return procs


def backend_process(listen_addr, cfg, identity):
    # Each backend process has a unique identity. For example, it can be responsible for rendering
    # a particular map. Ths backend process runs multiple subprocesses as workers and does
    # load-balancing.
    context = zmq.Context()

    frontend = context.socket(zmq.DEALER)
    frontend.setsockopt(zmq.IDENTITY, identity.encode())
    frontend.connect(listen_addr)

    backend = context.socket(zmq.DEALER)
    backend.setsockopt(zmq.RCVHWM, 0)
    backend.setsockopt(zmq.SNDHWM, 0)
    backend.setsockopt(zmq.LINGER, 0)

    backend_addr = 'ipc:///tmp/backendproc-%f' % (time.time())
    backend.bind(backend_addr)

    procs = start_generators(cfg['generator_cmds'][identity],
                             cfg.get('working_dir', './'),
                             cfg.get('envs', {}),
                             backend_addr)

    zmq.device(zmq.QUEUE, frontend, backend)


if __name__ == '__main__':
    def run():
        import tabulate
        import multiprocessing

        CONFIG_FILE = sys.argv[1]
        LISTEN_PORT = int(sys.argv[2])

        cfg = yaml.load(open(CONFIG_FILE).read(), Loader=yaml.SafeLoader)

        print('configurations:')
        print(tabulate.tabulate(cfg.items()))

        context = zmq.Context()

        frontend = context.socket(zmq.ROUTER)
        frontend.setsockopt(zmq.RCVHWM, 0)
        frontend.setsockopt(zmq.SNDHWM, 0)
        frontend.setsockopt(zmq.LINGER, 0)
        frontend.bind('tcp://*:%d' % LISTEN_PORT)

        backend = context.socket(zmq.ROUTER)
        backend.setsockopt(zmq.RCVHWM, 0)
        backend.setsockopt(zmq.SNDHWM, 0)
        backend.setsockopt(zmq.LINGER, 0)

        backend_addr = 'ipc:///tmp/backend-%s' % str(time.time())
        backend.bind(backend_addr)

        identities = cfg['generator_cmds'].keys()

        backend_procs = []

        for identity in identities:
            proc = multiprocessing.Process(
                target=backend_process,
                args=(backend_addr, cfg, identity))
            # proc.daemon = True
            proc.start()
            backend_procs.append(proc)

        poller = zmq.Poller()
        poller.register(backend, zmq.POLLIN)
        poller.register(frontend, zmq.POLLIN)

        identities = set([_.encode() for _ in identities])

        while True:
            socks = dict(poller.poll(None))
            if frontend in socks:
                try:
                    # dealer_id here is same as identity.
                    request_id, empty, dealer_id, request = frontend.recv_multipart()
                    if dealer_id not in identities:
                        print('nonexist identity:', dealer_id)
                        continue

                    # Note that there is no empty frame between dealer_id and request_id because
                    # backend will strip out dealer_id, leaving [request_id, '', request], which is
                    # a valid message for REP sockets.
                    backend.send_multipart([dealer_id, request_id, b'', request])
                except:
                    # This can happen when someone sends random requests to the listen port.
                    # We just ignore it.
                    print('frontend: error when receiving request. ignore the request.')
                    pass

            elif backend in socks:
                parts = backend.recv_multipart()
                dealer_id, request_id, empty, reply = parts
                frontend.send_multipart([request_id, b'', reply])

    run()
