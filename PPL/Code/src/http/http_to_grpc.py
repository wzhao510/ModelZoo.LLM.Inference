import argparse
import grpc
import llm_pb2
import llm_pb2_grpc
from flask import Flask, Response, request, jsonify
import msgpack

_app = Flask(__name__)
_grpc_server = ''

@_app.route("/ppl_llm_server", methods=['POST'])
def handle_post_req():
    req_data = request.get_json()
    channel = grpc.insecure_channel(_grpc_server)
    llmStub = llm_pb2_grpc.LLMServiceStub(channel)
    res_dict = {}
    grpc_req = llm_pb2.Request()
    req_list = llm_pb2.BatchedRequest()
    i = 1
    for req in req_data['prompt']:
        grpc_req.id = i
        res_dict[i] = ''
        i += 1
        grpc_req.temperature = req_data['temperature']
        grpc_req.generation_length = req_data['generation_length']
        grpc_req.prompt = req.encode()
        grpc_req.early_stopping=True
        req_list.req.append(grpc_req)
    responses = llmStub.Generation(req_list)

    def generate_chunks():
        for response in responses:
            for element in response.rsp:
                yield msgpack.packb({"id": element.id, "data": element.generated.decode("utf-8", "ignore")})

    return Response(generate_chunks(), mimetype='application/json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--threads")
    parser.add_argument("--grpc_server")
    args = parser.parse_args()
    _grpc_server = args.grpc_server
    from waitress import serve
    serve(_app, host=args.host, port=args.port, threads = args.threads)