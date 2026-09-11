"""Brokerless local ZMQ transport: PUB/SUB data and PUSH/PULL commands.

Each data topic has one publisher; multiple command clients connect to one
owner. Socket operations are serialized by a node lock, including shutdown.
Control acknowledgements use the result topic and request-id deduplication.
"""
from __future__ import annotations

import fcntl
import hashlib
import logging
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import zmq

from prometheus.transport.messages import Stamp, pack, unpack


def configure(raw=None):
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError("transport must be a mapping")
    if raw.get("backend", "zmq") != "zmq":
        raise ValueError("V3 uses transport.backend=zmq")
    if "namespace" in raw:
        value = str(raw["namespace"])
        if not value:
            raise ValueError("transport.namespace must be non-empty")
        os.environ["PROMETHEUS_ZMQ_NAMESPACE"] = value
    print(f"[transport] backend=zmq ipc namespace={os.environ.get('PROMETHEUS_ZMQ_NAMESPACE', 'prometheus-v3')}", flush=True)


def endpoint(domain, topic, *, namespace=None):
    namespace = namespace or os.environ.get("PROMETHEUS_ZMQ_NAMESPACE", "prometheus-v3")
    root = Path(f"/tmp/prometheus-zmq-{os.getuid()}")
    root.mkdir(mode=0o700, exist_ok=True)
    digest = hashlib.sha256(f"{namespace}\0{domain or 0}\0{topic}".encode()).hexdigest()[:32]
    return f"ipc://{root}/{digest}.sock"


class Context:
    def __init__(self, domain_id=None):
        self.domain_id = domain_id
        self._ok = True

    def ok(self):
        return self._ok

    def shutdown(self):
        self._ok = False


class _Clock:
    def now(self):
        stamp = time.time_ns()
        return SimpleNamespace(to_msg=lambda: Stamp(stamp // 1_000_000_000, stamp % 1_000_000_000))


class _Socket:
    def __init__(self, node, kind, topic, depth, *, bind):
        self.node = node
        self.address = endpoint(node.context.domain_id, topic)
        self.lock_file = None
        self.socket = node.zmq_context.socket(kind)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, max(1, int(depth)))
        self.socket.setsockopt(zmq.RCVHWM, max(1, int(depth)))
        self.socket.setsockopt(zmq.MAXMSGSIZE, 64 * 1024 * 1024)
        try:
            if kind == zmq.PUSH:
                self.socket.setsockopt(zmq.IMMEDIATE, 1)
            if bind:
                self.lock_file = open(self.address.removeprefix("ipc://") + ".lock", "a")
                try:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f"ZMQ endpoint already owned: domain={node.context.domain_id}, topic={topic}") from exc
                self.socket.bind(self.address)
            else:
                if kind == zmq.SUB:
                    self.socket.setsockopt(zmq.SUBSCRIBE, b"")
                self.socket.connect(self.address)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.socket.close(linger=0)
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None


class Publisher(_Socket):
    def __init__(self, node, topic, depth, *, command=False):
        self.command = command
        super().__init__(node, zmq.PUSH if command else zmq.PUB, topic, depth, bind=not command)

    def publish(self, message):
        payload = pack(message)
        with self.node.lock:
            self.socket.send(payload, flags=zmq.NOBLOCK)

    def get_subscription_count(self):
        # For control readiness, IMMEDIATE ensures POLLOUT only after connection.
        with self.node.lock:
            return int(bool(self.socket.poll(0, zmq.POLLOUT)))


class Node:
    def __init__(self, name, *, context=None, domain_id=None):
        self.name = name
        self.context = context or Context(domain_id)
        self.zmq_context = zmq.Context()
        self.lock = threading.RLock()
        self.publishers = []
        self.subscriptions = []
        self.timers = []
        self.closed = False

    def create_publisher(self, msg_type, topic, depth, *, command=False):
        with self.lock:
            pub = Publisher(self, topic, depth, command=command)
            self.publishers.append(pub)
            return pub

    def create_subscription(self, msg_type, topic, callback, depth, *, command=False):
        with self.lock:
            sub = _Socket(self, zmq.PULL if command else zmq.SUB, topic, depth, bind=command)
            self.subscriptions.append((sub, callback))
            return sub

    def create_timer(self, period, callback):
        timer = [float(period), time.monotonic() + float(period), callback]
        self.timers.append(timer)
        return timer

    def get_clock(self):
        return _Clock()

    def get_logger(self):
        return logging.getLogger(self.name)

    def spin_once(self, timeout_sec=0.0):
        with self.lock:
            if self.closed:
                return
            now = time.monotonic()
            timeout = max(0.0, float(timeout_sec))
            if self.timers:
                timeout = min(timeout, max(0.0, min(timer[1] for timer in self.timers) - now))
            poller = zmq.Poller()
            for sub, _ in self.subscriptions:
                poller.register(sub.socket, zmq.POLLIN)
            if self.subscriptions:
                ready = dict(poller.poll(max(0, math.ceil(timeout * 1000))))
                for sub, callback in self.subscriptions:
                    if sub.socket in ready:
                        callback(unpack(sub.socket.recv(flags=zmq.NOBLOCK)))
            elif timeout:
                time.sleep(timeout)
            now = time.monotonic()
            for timer in self.timers:
                if now >= timer[1]:
                    timer[1] = now + timer[0]  # no catch-up bursts after a slow callback
                    timer[2]()

    def destroy_node(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for resource in self.publishers + [sub for sub, _ in self.subscriptions]:
                resource.close()
            self.zmq_context.term()


class SingleThreadedExecutor:
    def __init__(self, context=None):
        self.context = context
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        self.nodes.remove(node)

    def spin_once(self, timeout_sec=0.0):
        for index, node in enumerate(self.nodes):
            node.spin_once(timeout_sec if index == 0 else 0.0)

    def shutdown(self):
        self.nodes.clear()
