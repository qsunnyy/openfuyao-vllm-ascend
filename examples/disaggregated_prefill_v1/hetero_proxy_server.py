# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.8+
# - Install dependencies:
#     pip install fastapi httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.


import argparse
import asyncio
import heapq
import os
import sys
from contextlib import asynccontextmanager
from typing import List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from vllm.logger import init_logger

logger = init_logger(__name__)

# Add uvloop for faster event loop if available
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerState:

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f'http://{host}:{port}/v1'
        self.client = httpx.AsyncClient(timeout=None,
                                        base_url=self.url,
                                        limits=httpx.Limits(
                                            max_connections=100000,
                                            max_keepalive_connections=100000))
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.aborted_requests = set()  # Track aborted requests
        # Removed individual server lock - will use global locks instead

class ProxyState:
    def __init__(self, grouped_instances):
        """
        grouped_instances: [
            {
                "prefillers": [(host, port, tp), ...],
                "decoders":   [(host, port), ...],
            },
            ...
        ]
        """
        self.groups = []
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        self.req_id_counter = 0

        for g in grouped_instances:
            group = {}
            group["prefillers"] = [ServerState(h, p) for h, p, t in g["prefillers"]]
            group["prefillers_tps"] = [t for h, p, t in g["prefillers"]]
            group["decoders"]   = [ServerState(h, p) for h, p in g["decoders"]]
            group["prefiller_heap"] = [(0, i, s) for i, s in enumerate(group["prefillers"])]
            group["decoder_heap"]   = [(0, i, s) for i, s in enumerate(group["decoders"])]
            heapq.heapify(group["prefiller_heap"])
            heapq.heapify(group["decoder_heap"])
            self.groups.append(group)

    def _update_prefiller_priority(self, group_id: int, server_idx: int):
        group = self.groups[group_id]
        server = group["prefillers"][server_idx]
        priority = server.active_tokens + server.active_kv_cache * 0.3
        group["prefiller_heap"] = [(p, i, s) for p, i, s in group["prefiller_heap"] if i != server_idx]
        heapq.heappush(group["prefiller_heap"], (priority, server_idx, server))

    def _update_decoder_priority(self, group_id: int, server_idx: int):
        group = self.groups[group_id]
        server = group["decoders"][server_idx]
        priority = server.active_tokens
        group["decoder_heap"] = [(p, i, s) for p, i, s in group["decoder_heap"] if i != server_idx]
        heapq.heappush(group["decoder_heap"], (priority, server_idx, server))

    def abort_prefiller_request(self, group_id: int, server_idx: int,
                                request_id):  # Changed to synchronous
        """
        Mark a request as aborted. This will helps to release kv cache in
        prefiller node.
        """
        # No lock needed - atomic operation
        group = self.groups[group_id]
        group["prefillers"][server_idx].aborted_requests.add(request_id)

    def aquire_aborted_prefiller_requests(
            self, group_id: int, server_idx: int):  # Changed to synchronous
        """
        Get the set of aborted requests and clear it.
        This is used to release kv cache in prefiller node.
        """
        # No lock needed - atomic operation
        group = self.groups[group_id]
        aborted_requests = group["prefillers"][server_idx].aborted_requests.copy()
        group["prefillers"][server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            self.req_id_counter += 1
            return str(self.req_id_counter)

    import numpy as np

    def select_group(self, request_length: int, alpha: float = 1.0, beta: float = 3.0,
                     bucket_seperate_length: int = 2048):
        n_groups = len(self.groups)
        tp_sums = np.array([sum(g["prefillers_tps"]) for g in self.groups], dtype=np.float64)
        active_tokens_sums = np.array([sum(s.active_tokens for s in g["prefillers"]) for g in self.groups],
                                      dtype=np.float64)

        total_tp_sum = tp_sums.sum()
        total_active_tokens = active_tokens_sums.sum() + 1e-6
        tp_factors =  tp_sums / total_tp_sum - 1 / n_groups
        tp_factors *= (1 - request_length / bucket_seperate_length)
        score_factors = alpha * (active_tokens_sums / total_active_tokens) + beta * tp_factors

        for gid, score in enumerate(score_factors):
            print(
                f"===Group {gid}: score={score:.6f}, request_length={request_length}, tp_sum={tp_sums[gid]}, active_tokens={active_tokens_sums[gid]}===")

        best_group_id = int(np.argmin(score_factors))
        print(f"===Selected group: {best_group_id}, min_score={score_factors[best_group_id]:.6f}===")
        return best_group_id


    def select_prefiller(self, group_id: int, token_count: int):
        group = self.groups[group_id]
        if not group["prefiller_heap"]:
            raise RuntimeError("No prefiller servers available in group")

        priority, chosen, server = heapq.heappop(group["prefiller_heap"])
        server.active_tokens += token_count
        server.active_kv_cache += token_count
        self._update_prefiller_priority(group_id, chosen)
        return chosen

    def release_prefiller(self, group_id: int, idx: int, token_count: int):
        group = self.groups[group_id]
        group["prefillers"][idx].active_tokens -= token_count
        self._update_prefiller_priority(group_id, idx)

    def release_prefiller_kv(self, group_id: int, idx: int, token_count: int):
        group = self.groups[group_id]
        if group["prefillers"][idx].active_kv_cache > 0:
            group["prefillers"][idx].active_kv_cache -= token_count
        self._update_prefiller_priority(group_id, idx)

    def select_decoder(self, group_id: int, token_count: int):
        group = self.groups[group_id]
        if not group["decoder_heap"]:
            raise RuntimeError("No decoder servers available in group")

        priority, chosen, server = heapq.heappop(group["decoder_heap"])
        server.active_tokens += token_count
        self._update_decoder_priority(group_id, chosen)
        return chosen

    def release_decoder(self, group_id: int, idx: int, token_count: int):
        group = self.groups[group_id]
        group["decoders"][idx].active_tokens -= token_count
        self._update_decoder_priority(group_id, idx)

    # request_length = 10 → 120.16
    # request_length = 100 → 120.94
    # request_length = 1000 → 128.70
    # request_length = 5000 → 163.20
    # request_length = 10000 → 206.32
    # request_length = 20000 → 292.57
    # 分数函数保持不变
    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        return length_score * 0.0345 + 120.0745

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length


proxy_state = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        type=int,
                        nargs="+",
                        default=[8001])
    parser.add_argument("--prefiller-tps",
                        type=int,
                        nargs="+",
                        default=[1])
    parser.add_argument("--decoder-hosts",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries",
                        type=int,
                        default=3,
                        help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.001,
        help="Base delay (seconds) for exponential backoff retries")
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")
    if len(args.prefiller_hosts) != len(args.prefiller_tps):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller tp config")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")
    # args.prefiller_instances = list(
    #     zip(args.prefiller_hosts, args.prefiller_ports, args.prefiller_tps))
    # args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    args.grouped_instances = [
        {
            "prefillers": [(h, p, tp)],
            "decoders": [(dh, dp)]
        }
        for (h, p, tp), (dh, dp) in zip(
            zip(args.prefiller_hosts, args.prefiller_ports, args.prefiller_tps),
            zip(args.decoder_hosts, args.decoder_ports)
        )
    ]
    
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state

    print("----------------------------global_argsgrouped_instances:",global_args.grouped_instances)
    
    proxy_state = ProxyState(global_args.grouped_instances)
    print(
        f"Initialized {len(proxy_state.groups)} pd groups"
    )
    yield
    # 正确关闭 client
    for group in proxy_state.groups:
        for p in group["prefillers"]:
            await p.client.aclose()
        for d in group["decoders"]:
            await d.client.aclose()


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(client: httpx.AsyncClient,
                                  group_id: int,
                                  prefiller_id: int,
                                  endpoint: str,
                                  req_data: dict,
                                  request_id: str,
                                  max_retries: int = 1,
                                  base_delay: float = 0.2):
    aborted_requests = proxy_state.aquire_aborted_prefiller_requests(
        group_id, prefiller_id)
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        #"do_remote_prefill": True,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_requests),
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint,
                                         json=req_data,
                                         headers=headers,
                                         timeout=5.0)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(
                f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2**(attempt - 1)))
            else:
                logger.error(
                    f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc


async def stream_service_response_with_retry(client: httpx.AsyncClient,
                                             endpoint: str,
                                             req_data: dict,
                                             request_id: str,
                                             max_retries: int = 3,
                                             base_delay: float = 0.2):
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST",
                                     endpoint,
                                     json=req_data,
                                     headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return  # Success, exit after streaming
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}"
                )
                await asyncio.sleep(base_delay * (2**(attempt - 1)))
            else:
                logger.error(
                    f"All {max_retries} attempts failed for streaming {endpoint}."
                )
                raise e
        except Exception as e:
            # If any chunk has been sent, do not retry, just log and drop
            if 'first_chunk_sent' in locals() and first_chunk_sent:
                logger.error(
                    f"Streaming to client interrupted after response started: {str(e)}"
                )
                return
            else:
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}"
                    )
                    await asyncio.sleep(base_delay * (2**(attempt - 1)))
                else:
                    logger.error(
                        f"All {max_retries} attempts failed for streaming {endpoint}."
                    )
                    raise e

async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        prefiller_score = proxy_state.calculate_prefill_scores(request_length)
        request_id = await proxy_state.next_req_id()

        # 选择hetero group
        group_id = proxy_state.select_group(request_length)

        # 选择 prefiller
        prefiller_idx = proxy_state.select_prefiller(group_id, prefiller_score)
        prefiller = proxy_state.groups[group_id]["prefillers"][prefiller_idx]

        if "max_completion_tokens" in req_data:
            max_completion_tokens = req_data.pop("max_completion_tokens")
            req_data["max_tokens"] = max_completion_tokens
        print(f"############ req_data: {req_data} ###############")

        response = await send_request_to_service(
            prefiller.client,
            group_id,
            prefiller_idx,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay
        )
        proxy_state.release_prefiller(group_id, prefiller_idx, prefiller_score)

        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # ⚡ 选择 decoder 同组
        decoder_score = proxy_state.calculate_decode_scores(request_length)
        decoder_idx = proxy_state.select_decoder(group_id, decoder_score)
        decoder = proxy_state.groups[group_id]["decoders"][decoder_idx]

        async def generate_stream():
            released_kv = False
            try:
                async for chunk in stream_service_response_with_retry(
                    decoder.client,
                    api,
                    req_data,
                    request_id=request_id,
                    max_retries=global_args.max_retries,
                    base_delay=global_args.retry_delay
                ):
                    if not released_kv and chunk:
                        proxy_state.release_prefiller_kv(group_id, prefiller_idx, prefiller_score)
                        released_kv = True
                    yield chunk
            finally:
                proxy_state.release_decoder(group_id, decoder_idx, decoder_score)

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              f" - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise

@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders)
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
