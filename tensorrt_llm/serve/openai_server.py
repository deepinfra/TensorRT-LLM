#!/usr/bin/env python
import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, List, Optional, Tuple, TypedDict, Callable, Awaitable, Any, Type

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.utils import nvtx_mark
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                ChatCompletionResponse,
                                                CompletionRequest,
                                                CompletionResponse,
                                                CompletionResponseChoice,
                                                ErrorResponse, ModelCard,
                                                ModelList, UsageInfo)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs, CompletionPostprocArgs, chat_response_post_processor,
    chat_stream_post_processor, completion_response_post_processor,
    completion_stream_post_processor)
from tensorrt_llm.version import __version__ as VERSION
from tensorrt_llm._torch.pyexecutor.py_executor import PROM_METRICS_FILENAME

from collections import defaultdict
import array
import json
import os
import traceback

prom_metrics_file = None
prom_metrics = defaultdict(float, {
    "num_requests_running": 0,
    "num_requests_waiting": 0,
    "prompt_tokens_total": 0,
    "generation_tokens_total": 0,
})

# yapf: enale
TIMEOUT_KEEP_ALIVE = 5  # seconds.


async def disconnect_poller(request: Request, result: Any):
    """
    Poll for a disconnect.
    If the request disconnects, stop polling and return.
    """
    try:
        # while not await request.is_disconnected():
        #     await asyncio.sleep(0.01)
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                break

        print("Request disconnected")

        return result
    except asyncio.CancelledError:
        print("Stopping polling loop")


def cancel_on_disconnect(model_type: Type[BaseModel]):
    """
    Decorator that will check if the client disconnects,
    and cancel the task if required.
    """

    def cancel_on_disconnect_inner(handler: Callable):

        @wraps(handler)
        async def cancel_on_disconnect_decorator(self, fastapi_request: Request, request: model_type):
            sentinel = object()
            print(self, fastapi_request, request)

            # Create two tasks, one to poll the request and check if the
            # client disconnected, and another which is the request handler
            poller_task = asyncio.ensure_future(disconnect_poller(fastapi_request, sentinel))
            handler_task = asyncio.ensure_future(handler(self, fastapi_request=fastapi_request, request=request))

            done, pending = await asyncio.wait(
                [poller_task, handler_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel any outstanding tasks
            for t in pending:
                t.cancel()

                try:
                    await t
                except asyncio.CancelledError:
                    print(f"{t} was cancelled")
                except Exception as exc:
                    print(f"{t} raised {exc} when being cancelled")

            # Return the result if the handler finished first
            if handler_task in done:
                return await handler_task

            # Otherwise, raise an exception
            # This is not exactly needed, but it will prevent
            # validation errors if your request handler is supposed
            # to return something.
            print("Raising an HTTP error because I was disconnected!!")

            raise HTTPException(503)

        return cancel_on_disconnect_decorator

    return cancel_on_disconnect_inner


class ConversationMessage(TypedDict):
    role: str
    content: str


def parse_chat_message_content(
    message: ChatCompletionMessageParam, ) -> ConversationMessage:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return []
    if isinstance(content, str):
        return [ConversationMessage(role=role, content=content)]

    # for Iterable[ChatCompletionContentPartTextParam]
    texts: List[str] = []
    for part in content:
        part_type = part["type"]
        if part_type == "text":
            text = part["text"]
            texts.append(text)
        else:
            raise NotImplementedError(f"{part_type} is not supported")

    text_prompt = "\n".join(texts)
    return [ConversationMessage(role=role, content=text_prompt)]


class OpenAIServer:

    def __init__(self,
                 llm: LLM,
                 model: str,
                 served_model_name: str | None,
                 hf_tokenizer: PreTrainedTokenizer = None):
        self.llm = llm
        self.tokenizer = hf_tokenizer

        model_dir = Path(model)
        if served_model_name is not None:
            self.model = served_model_name
        elif model_dir.exists() and model_dir.is_dir():
            self.model = model_dir.name
        else:
            self.model = model

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # terminate rank0 worker
            yield
            self.llm.shutdown()

        self.app = FastAPI(lifespan=lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            return self.create_error_response(message=str(exc))

        self.register_routes()

    @property
    def postproc_worker_enabled(self) -> bool:
        return True if self.llm.args._num_postprocess_workers > 0 else False

    @staticmethod
    def create_error_response(
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        error_response = ErrorResponse(message=message,
                                       type=err_type,
                                       code=status_code.value)
        return JSONResponse(content=error_response.model_dump(),
                            status_code=error_response.code)

    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/metrics", self.metrics, methods=["GET"])
        self.app.add_api_route("/metrics/", self.metrics, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        self.app.add_api_route("/v1/completions",
                               self.openai_completion,
                               methods=["POST"])
        self.app.add_api_route("/v1/chat/completions",
                               self.openai_chat,
                               methods=["POST"])

    async def health(self) -> Response:
        return Response(status_code=200)

    async def version(self) -> JSONResponse:
        ver = {"version": VERSION}
        return JSONResponse(content=ver)

    async def metrics(self) -> Response:
        global prom_metrics_file
        bufs = None
        try:
            if prom_metrics_file is None:
                prom_metrics_file = os.open(PROM_METRICS_FILENAME,
                                            os.O_RDWR|os.O_CREAT|os.O_TRUNC)
            bufs = os.pread(prom_metrics_file, 65536, 0).split(b'\0', 1)
            if len(bufs) >= 2:
                keybuf, valbuf = bufs
                key_list = json.loads(keybuf.decode('UTF-8'))
                value_list = array.array('d')
                value_list.frombytes(valbuf)
                for key, value in zip(key_list, value_list):
                    prom_metrics[key] = value
        except:
            print(bufs)
            traceback.print_exc()

        all_requests_done = (
                prom_metrics["request_completed_total"] +
                prom_metrics["request_cancelled_total"] +
                prom_metrics["request_failed_total"])
        # NOTE: metrics do not update if the other thread is not running any requests.
        # Make sure to zero out running and waiting in this case.
        if prom_metrics["request_started_total"] == all_requests_done:
            prom_metrics["num_requests_running"] = 0

        # Detect number of requests not being processed by the TensorRT-LLM engine.
        prom_metrics["num_requests_waiting"] = max(0, prom_metrics["request_started_total"] - (
                prom_metrics["num_requests_running"] + all_requests_done))

        resp = ''
        for metric_key, metric_val in prom_metrics.items():
            separator = ',' if '{' in metric_key else '{'
            resp += f'vllm:{metric_key}{separator}model_name="{self.model}"}} {float(metric_val)}\n'
        return Response(status_code=200, content=resp)

    async def get_model(self) -> JSONResponse:
        model_list = ModelList(data=[ModelCard(id=self.model)])
        return JSONResponse(content=model_list.model_dump())

    @cancel_on_disconnect(ChatCompletionRequest)
    async def openai_chat(self, fastapi_request: Request, request: ChatCompletionRequest) -> Response:

        did_complete = False

        def get_role() -> str:
            if request.add_generation_prompt:
                role = "assistant"
            else:
                role = request.messages[-1]["role"]
            return role

        async def chat_stream_generator(
                promise: RequestOutput, postproc_params: PostprocParams) -> AsyncGenerator[str, None]:
            nonlocal did_complete
            if not self.postproc_worker_enabled:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
            try:
                async for res in promise:
                    pp_results = res.outputs[0]._postprocess_result if self.postproc_worker_enabled else post_processor(res, args)
                    for pp_res in pp_results:
                        for choice in pp_res.choices:
                            if choice.finish_reason is not None:
                                did_complete = True
                                prom_metrics["request_completed_total"] += 1
                                prom_metrics[f"request_success_total{{finished_reason=\"{choice.finish_reason}\""] += 1

                        pp_res_json = pp_res.model_dump_json(exclude_unset=True)
                        yield f"data: {pp_res_json}\n\n"
                yield f"data: [DONE]\n\n"
                nvtx_mark("generation ends")
            finally:
                if not did_complete:
                    prom_metrics["request_cancelled_total"] += 1
                    promise.abort()

        async def create_chat_response(
                promise: RequestOutput, postproc_params: PostprocParams) -> ChatCompletionResponse:
            await promise.aresult()
            if self.postproc_worker_enabled:
                return promise.outputs[0]._postprocess_result
            else:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                res = post_processor(promise, args)
                for choice in res.choices:
                    if choice.finish_reason is not None:
                        prom_metrics["request_completed_total"] += 1
                        prom_metrics[f"request_success_total{{finished_reason=\"{choice.finish_reason}\""] += 1


        prom_metrics["request_started_total"] += 1
        promise: Optional[RequestOutput] = None
        try:
            conversation: List[ConversationMessage] = []
            for msg in request.messages:
                conversation.extend(parse_chat_message_content(msg))
            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]
            prompt: str = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                **(request.chat_template_kwargs or {}),
            )
            sampling_params = request.to_sampling_params()
            postproc_args = ChatPostprocArgs.from_request(request)
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == get_role():
                postproc_args.last_message_content = conversation[-1]["content"]
            postproc_params = PostprocParams(
                post_processor=chat_stream_post_processor
                if request.stream else chat_response_post_processor,
                postproc_args=postproc_args,
            )

            promise = self.llm.generate_async(
                inputs=prompt,
                sampling_params=sampling_params,
                _postproc_params=postproc_params if self.postproc_worker_enabled else None,
                streaming=request.stream,
            )
            if not self.postproc_worker_enabled:
                postproc_args.tokenizer = self.tokenizer
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            if request.stream:
                response_generator = chat_stream_generator(promise, postproc_params)
                return StreamingResponse(content=response_generator,
                                         media_type="text/event-stream")
            else:
                response = await create_chat_response(promise, postproc_params)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except asyncio.CancelledError:
            prom_metrics["request_cancelled_total"] += 1
            if promise is not None:
                promise.abort()
            return self.create_error_response("cancelled")
        except Exception as e:
            prom_metrics["request_failed_total"] += 1
            return self.create_error_response(str(e))

    @cancel_on_disconnect(CompletionRequest)
    async def openai_completion(self, fastapi_request: Request, request: CompletionRequest) -> Response:

        def merge_promises(
            promises: List[RequestOutput],
            postproc_params_collections: List[Optional[PostprocParams]]
        ) -> AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]:
            outputs = asyncio.Queue()
            finished = [False] * len(promises)

            async def producer(i: int, promise: RequestOutput, postproc_params: Optional[PostprocParams]):
                async for output in promise:
                    await outputs.put((output, postproc_params))
                finished[i] = True

            _tasks = [
                asyncio.create_task(producer(i, promise, postproc_params))
                for i, (promise, postproc_params) in enumerate(zip(promises, postproc_params_collections))
            ]

            async def consumer():
                while not all(finished) or not outputs.empty():
                    item = await outputs.get()
                    yield item
                await asyncio.gather(*_tasks)

            return consumer()

        async def create_completion_generator(
                generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]],
                promises: List[RequestOutput]):
            did_complete = False
            try:
                async for request_output, postproc_params in generator:
                    if not self.postproc_worker_enabled:
                        post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                        pp_result = post_processor(request_output, args)
                    else:
                        pp_result = request_output.outputs[0]._postprocess_result
                    for pp_res in pp_result:
                        for choice in pp_res.choices:
                            if choice.finish_reason is not None:
                                did_complete = True
                                prom_metrics["request_completed_total"] += 1
                                prom_metrics[f"request_success_total{{finished_reason=\"{choice.finish_reason}\""] += 1
                        pp_res_json = pp_res.model_dump_json(exclude_unset=False)
                        yield f"data: {pp_res_json}\n\n"
                yield f"data: [DONE]\n\n"
            finally:
                print(f"Completion generator finally {did_complete=}")
                if not did_complete:
                    prom_metrics["request_cancelled_total"] += 1
                    for promise in promises:
                        promise.abort()

        async def create_completion_response(
                generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]) -> CompletionResponse:
            all_choices: List[CompletionResponseChoice] = []
            num_prompt_tokens = num_gen_tokens = 0
            async for request_output, postproc_params in generator:
                pp_result: CompletionResponse
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                    pp_result = post_processor(request_output, args)
                else:
                    pp_result = request_output.outputs[0]._postprocess_result

                for choice in pp_result.choices:
                    if choice.finish_reason is not None:
                        prom_metrics["request_completed_total"] += 1
                        prom_metrics[f"request_success_total{{finished_reason=\"{choice.finish_reason}\""] += 1

                choices, usage = pp_result.choices, pp_result.usage
                all_choices.extend(choices)
                num_prompt_tokens += usage.prompt_tokens
                num_gen_tokens += usage.completion_tokens

            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_gen_tokens,
                total_tokens=num_gen_tokens + num_prompt_tokens,
            )
            response = CompletionResponse(
                model=self.model,
                choices=all_choices,
                usage=usage_info,
            )
            return response

        prom_metrics["request_started_total"] += 1
        try:
            if isinstance(request.prompt, str) or \
                (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
                prompts = [request.prompt]
            else:
                prompts = request.prompt

            promises: List[RequestOutput] = []
            postproc_params_collection: List[Optional[PostprocParams]] = []
            sampling_params = request.to_sampling_params()
            disaggregated_params = request.to_llm_disaggregated_params()
            for idx, prompt in enumerate(prompts):
                postproc_args = CompletionPostprocArgs.from_request(request)
                postproc_args.prompt_idx = idx
                if request.echo:
                    postproc_args.prompt = prompt
                postproc_params = PostprocParams(
                    post_processor=completion_stream_post_processor
                    if request.stream else completion_response_post_processor,
                    postproc_args=postproc_args,
                )
                promise = self.llm.generate_async(
                    inputs=prompt,
                    sampling_params=sampling_params,
                    _postproc_params=postproc_params,
                    streaming=request.stream,
                    disaggregated_params=disaggregated_params
                )
                if not self.postproc_worker_enabled:
                    postproc_args.tokenizer = self.tokenizer
                    postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)
                promises.append(promise)
                postproc_params_collection.append(None if self.postproc_worker_enabled else postproc_params)

            generator = merge_promises(promises, postproc_params_collection)
            if request.stream:
                response_generator = create_completion_generator(generator, promises)
                return StreamingResponse(content=response_generator,
                                            media_type="text/event-stream")
            else:
                response = await create_completion_response(
                    generator)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except asyncio.CancelledError:
            prom_metrics["request_cancelled_total"] += 1
            for promise in promises:
                promise.abort()
            return self.create_error_response("cancelled")
        except Exception as e:
            print(f"Encountered an exception: {str(e)}")
            prom_metrics["request_failed_total"] += 1
            return self.create_error_response(str(e))

    async def __call__(self, host, port):
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()
