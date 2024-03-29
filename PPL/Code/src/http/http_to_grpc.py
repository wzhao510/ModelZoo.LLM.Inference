import argparse
import grpc
import llm_pb2
import llm_pb2_grpc
from flask import Flask, request, jsonify
from flask_socketio import SocketIO

_app = Flask(__name__)
socketio = SocketIO(_app, async_mode='threading')
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
    for req in req_data['data']:
        grpc_req.id = i
        res_dict[i] = ''
        i += 1
        grpc_req.temperature = req_data['temperature']
        grpc_req.generation_length = req_data['generation_length']
        grpc_req.prompt = req
        req_list.req.append(grpc_req)
    responses = llmStub.Generation(req_list)
    for response in responses:
        res_dict[response.id] += response.generated
    resp_data = {'code': 0, 'data': list(res_dict.values())}
    return jsonify(resp_data), 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--grpc_server")
    args = parser.parse_args()
    _grpc_server = args.grpc_server
    #_app.run(debug=True, host=args.host, port=int(args.port))
    socketio.run(_app, host=args.host, port=int(args.port))
