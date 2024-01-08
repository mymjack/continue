import json
from typing import Any, Dict, Optional

from pydantic import Field, validator

from .base import LLM

import boto3
import json
import io

sagemaker = boto3.client("sagemaker-runtime", region_name='us-east-1')


class LlamaCpp(LLM):
    """
    Run the llama.cpp server binary to start the API server. If running on a remote server, be sure to set host to 0.0.0.0:

    ```shell
    .\\server.exe -c 4096 --host 0.0.0.0 -t 16 --mlock -m models\\meta\\llama\\codellama-7b-instruct.Q8_0.gguf
    ```

    After it's up and running, change `~/.continue/config.json` to look like this:

    ```json title="~/.continue/config.json"
    {
        "models": [{
            "title": "Llama CPP",
            "provider": "llama.cpp",
            "model": "MODEL_NAME",
            "sagemaker_endpoint_name": "http://localhost:8080"
        }]
    }
    ```
    """

    model: str = "llamacpp"
    sagemaker_endpoint_name: Optional[str] = Field(
        "CodeLlama7BInstructStreaming", description="Name of the SageMaker endpoint"
    )

    @validator("sagemaker_endpoint_name", pre=True, always=True)
    def set_sagemaker_endpoint_name(cls, sagemaker_endpoint_name):
        return sagemaker_endpoint_name or "CodeLlama7BInstructStreaming"

    llama_cpp_args: Dict[str, Any] = Field(
        {"stop": ["[INST]"]},
        description="A list of additional arguments to pass to llama.cpp. See https://github.com/ggerganov/llama.cpp/tree/master/examples/server#api-endpoints for the complete catalog of options.",
    )

    class Config:
        arbitrary_types_allowed = True

    def collect_args(self, options) -> Any:
        args = super().collect_args(options)
        if "max_tokens" in args:
            args["n_predict"] = args["max_tokens"]
            del args["max_tokens"]
        if "frequency_penalty" in args:
            del args["frequency_penalty"]
        if "presence_penalty" in args:
            del args["presence_penalty"]

        for k, v in self.llama_cpp_args.items():
            if k not in args:
                args[k] = v

        return args

    async def _stream_complete(self, prompt, options):
        args = self.collect_args(options)

        async def server_generator():
            # api = "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/CodeLlama7BInstruct/invocations"
            smr_inference_stream = SmrInferenceStream(
                sagemaker, self.sagemaker_endpoint_name)
            stream = smr_inference_stream.stream_inference({
                "action": "completion",
                "completion": {
                    "stream": True,
                    "prompt": prompt,
                    "temperature": 0.10,
                    "max_tokens": 600,
                    "top_p": 0.90,
                    "stop": [
                        "\n\n\n\n",
                        "[INST]",
                        "[/CODE]"
                    ]
                }
            })
            for chunk in stream:
                yield chunk

        async for chunk in server_generator():
            yield chunk


class SmrInferenceStream:
    def __init__(self, sagemaker_runtime, endpoint_name):
        self.sagemaker_runtime = sagemaker_runtime
        self.endpoint_name = endpoint_name
        # A buffered I/O stream to combine the payload parts:
        self.buff = io.BytesIO()
        self.read_pos = 0

    def stream_inference(self, request_body):
        # Gets a streaming inference response
        # from the specified model endpoint:
        response = self.sagemaker_runtime \
            .invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name,
            Body=json.dumps(request_body),
            ContentType="application/json"
        )
        # Gets the EventStream object returned by the SDK:
        event_stream = response['Body']
        for event in event_stream:
            # Passes the contents of each payload part
            # to be concatenated:
            self._write(event['PayloadPart']['Bytes'])
            # Iterates over lines to parse whole JSON objects:
            for line in self._readlines():
                if len(line) == 0:
                    continue
                resp = json.loads(line)
                part = resp.get("choices")[0]['text']
                if part != "":
                    yield part
        # yield ' <END>'

    # Writes to the buffer to concatenate the contents of the parts:
    def _write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)

    # The JSON objects in buffer end with '\n'.
    # This method reads lines to yield a series of JSON objects:
    def _readlines(self):
        self.buff.seek(self.read_pos)
        delimiter = b"*nestW0rd*"
        content = self.buff.read().decode('utf-8')  # Decode bytes to string
        lines = content.split(delimiter.decode('utf-8'))
        for line in lines[
                    :-1]:  # Exclude the last element as it may be an empty string the add delimiter will cause wrong offset
            self.read_pos += (len(line) + len(delimiter))
            yield line



