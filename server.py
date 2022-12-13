from http.server import HTTPServer
from http.server import BaseHTTPRequestHandler
import json

class MyRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        # self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

        file_content = open("input.html", "r").read()
        self.wfile.write(file_content.encode("utf-8"))


    def do_POST(self):
        datalen = int(self.headers['Content-Length'])
        data = self.rfile.read(datalen)
        obj = json.loads(data)
        print("Got object: {}".format(obj))

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        # self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

        self.wfile.write("<html><head><title>Title goes here.</title></head></html>".encode("utf-8"))

server = HTTPServer(('', 12345), MyRequestHandler)
server.serve_forever()