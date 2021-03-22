import zmq
import sys
import gflags
import time
import multiprocessing
import threading
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
from .inference import find_tracker


gflags.DEFINE_string('addr', '', '')
gflags.DEFINE_string('weights_file', '', '')
gflags.DEFINE_integer('n_worker', 1, '')
gflags.DEFINE_boolean('with_id', False, '')
gflags.DEFINE_string('devices', 'cuda', 'Comma separated list of device.')


FLAGS = gflags.FLAGS
FLAGS(sys.argv)


def backend_process(listen_addr, device, identity):
    context = zmq.Context()

    if identity is not None:
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, identity)
    else:
        socket = context.socket(zmq.REP)

    socket.connect(listen_addr)

    # Weights file can be empty string. If that's the case, the first command received must be "reload"
    # which actually loads the weights.
    tracker = None
    if FLAGS.weights_file != '':
        tracker = find_tracker(FLAGS.weights_file)(FLAGS.weights_file, device)

    def exec_cmd(msg):
        func_name = msg[0]
        args = msg[1:]
        try:
            ret = getattr(tracker, func_name)(*args)
            return ret
        except:
            print('Failed to execute message: %r' % msg)
            raise

    should_exit = False
    while not should_exit:
        if identity is not None:
            request_id, _, msg = socket.recv_multipart()
        else:
            msg = socket.recv()

        msg = msgpack.unpackb(msg, raw=False)
        ret = None

        if msg[0] == 'exit':
            ret = b'ok'
            should_exit = True

        elif msg[0] == 'cmd_list':
            for m in msg[1:]:
                ret = exec_cmd(m)

        elif msg[0] == 'reload':
            weights_file = msg[1]
            if tracker is None:
                tracker = find_tracker(weights_file)(weights_file, device)
            else:
                tracker.reload(weights_file)

        else:
            ret = exec_cmd(msg)

        rep = msgpack.packb(ret, use_bin_type=True)
        if identity is not None:
            socket.send_multipart([request_id, b'', rep])
        else:
            socket.send(rep)

    socket.disconnect(listen_addr)
    socket.close()
    context.term()
    print('progress tracker inference server backend terminated')


if __name__ == '__main__':
    def run():
        context = zmq.Context()

        print('inference server addr', FLAGS.addr)

        frontend = context.socket(zmq.ROUTER)
        frontend.setsockopt(zmq.RCVHWM, 0)
        frontend.setsockopt(zmq.SNDHWM, 0)
        frontend.setsockopt(zmq.LINGER, 0)
        frontend.bind(FLAGS.addr)

        backend = context.socket(zmq.ROUTER)
        backend.setsockopt(zmq.RCVHWM, 0)
        backend.setsockopt(zmq.SNDHWM, 0)
        backend.setsockopt(zmq.LINGER, 0)

        backend_addr = 'ipc:///tmp/progress_tracker_inference-backend-%s' % str(time.time())
        backend.bind(backend_addr)

        backend_procs = []

        devices = FLAGS.devices.split(',')

        def run_no_id():
            for i in range(FLAGS.n_worker):
                proc = multiprocessing.Process(target=backend_process, args=(backend_addr, devices[i % len(devices)], None))
                proc.start()
                backend_procs.append(proc)

            def monitor_backend():
                print('monitor thread started')
                for proc in backend_procs:
                    proc.join()
                print('all progress tracker backend procs terminated')
                context.term()

            monitor_thread = threading.Thread(target=monitor_backend)
            monitor_thread.start()

            try:
                zmq.device(zmq.QUEUE, frontend, backend)
            except zmq.error.ContextTerminated:
                frontend.close()
                backend.close()
                print('progress tracker inference server frontend terminated')

        def run_with_id():
            identities = [b'tracker-%d' % _ for _ in range(FLAGS.n_worker)]
            for i in range(FLAGS.n_worker):
                proc = multiprocessing.Process(target=backend_process, args=(backend_addr, devices[i % len(devices)],
                                                                             identities[i]))
                proc.start()
                backend_procs.append(proc)

            poller = zmq.Poller()
            poller.register(backend, zmq.POLLIN)
            poller.register(frontend, zmq.POLLIN)

            while True:
                socks = dict(poller.poll(None))
                if frontend in socks:
                    try:
                        # dealer_id here is same as identity.
                        parts = frontend.recv_multipart()
                        request_id, empty, dealer_id, request = parts
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

        if FLAGS.with_id:
            run_with_id()
        else:
            run_no_id()

    run()
