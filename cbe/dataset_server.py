import zmq
import sys
import gflags
import time
import yaml
import multiprocessing
import threading
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from cbe.inference import Client
import cbe.dataset


gflags.DEFINE_string('addr', '', '')
gflags.DEFINE_integer('n_worker', 1, 'Number of dataset workers.')
gflags.DEFINE_integer('n_tracker', 1, 'Number of tracker servers.')
gflags.DEFINE_string('class_name', '', '')
gflags.DEFINE_string('devices', 'cuda', 'Comma separated list of devices')
gflags.DEFINE_string('param_file', '', 'A Yaml file containing arguments for constructing the dataset')


FLAGS = gflags.FLAGS
FLAGS(sys.argv)

dataset_class = getattr(cbe.dataset, FLAGS.class_name)


def backend_process(listen_addr, dataset, worker_id, tracker_id):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(listen_addr)

    dataset.worker_id = worker_id
    dataset.tracker_id = tracker_id

    def exec_cmd(msg):
        func_name = msg[0]
        args = msg[1:]
        try:
            ret = getattr(dataset, func_name)(*args)
            return ret
        except:
            print('Failed to execute message: %r' % msg)
            raise

    idx = 0
    while True:
        msg = msgpack.unpackb(socket.recv(), raw=False)
        print('received', idx)
        ret = None

        if msg[0] == 'exit':
            socket.send_pyobj('ok')
            break

        elif msg[0] == 'cmd_list':
            for m in msg[1:]:
                ret = exec_cmd(m)
        else:
            ret = exec_cmd(msg)

        print('before sending', idx)
        socket.send(msgpack.packb(ret, use_bin_type=True), flags=zmq.NOBLOCK)
        print('sent', idx)
        idx += 1

    socket.disconnect(listen_addr)
    socket.close()
    context.term()
    print('progress tracker dataset server backend terminated')


if __name__ == '__main__':
    context = zmq.Context()

    frontend = context.socket(zmq.ROUTER)
    frontend.setsockopt(zmq.RCVHWM, 0)
    frontend.setsockopt(zmq.SNDHWM, 0)
    frontend.setsockopt(zmq.LINGER, 0)
    frontend.bind(FLAGS.addr)

    backend = context.socket(zmq.DEALER)
    backend.setsockopt(zmq.RCVHWM, 0)
    backend.setsockopt(zmq.SNDHWM, 0)
    backend.setsockopt(zmq.LINGER, 0)

    backend_addr = 'ipc:///tmp/cbe_dataset-backend-%s' % str(time.time())
    backend.bind(backend_addr)

    backend_procs = []

    kwargs = yaml.load(open(FLAGS.param_file).read(), Loader=yaml.SafeLoader)

    devices = FLAGS.devices.split(',')

    tracker_server_port = 5003  # FIXME: remove hardcoded value

    # Launch tracker servers.
    # Note that weights_files are not specified.
    tracker_server_proc = Client.launch_server(
        '', 'tcp://*:%d' % tracker_server_port, FLAGS.devices, FLAGS.n_tracker, with_id=True)

    dataset = dataset_class(device=devices[0],
                            tracker_server_addr='tcp://localhost:%d' % tracker_server_port,
                            **kwargs)

    tracker_ids = [b'tracker-%d' % _ for _ in range(FLAGS.n_tracker)]

    for i in range(FLAGS.n_worker):
        proc = multiprocessing.Process(target=backend_process,
                                       args=(backend_addr, dataset, i, tracker_ids[i % FLAGS.n_tracker]))
        proc.start()
        backend_procs.append(proc)

    def monitor_backend():
        print('monitor thread started')
        for proc in backend_procs:
            proc.join()
        print('all progress tracker dataset backend procs terminated')
        context.term()

    monitor_thread = threading.Thread(target=monitor_backend)
    monitor_thread.start()

    try:
        zmq.device(zmq.QUEUE, frontend, backend)
    except zmq.error.ContextTerminated:
        frontend.close()
        backend.close()
        print('progress tracker dataset server frontend terminated')
